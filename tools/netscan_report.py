"""Summarises the packet capture of a container build: every outbound connection and the host name it used.

  python3 -I tools/netscan_report.py build.pcap [build.log] > report.md

Reads the capture that tools/build_netscan.sh writes (tcpdump -w; Linux cooked v1/v2 or Ethernet). Prints Markdown:
one row per host name (TLS SNI, HTTP Host header, or else the DNS name that resolved to the address) with its ports,
addresses and number of new connections; HTTP request lines; and, from the build log, what the build fetched (base
images, Ubuntu packages, git clones and their commits, Python packages). Standard library only. Reads only.
"""
import collections, datetime, ipaddress, re, struct, sys

LINK_HEADER = {1: 14, 113: 16, 276: 20}   # Ethernet, Linux cooked v1, Linux cooked v2


def packets(path):
    """(time, IP packet bytes) for each IPv4/IPv6 packet in a pcap file."""
    data = open(path, "rb").read()
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        end = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        end = ">"
    else:
        sys.exit(f"{path}: not a pcap file (pcapng is not read; use tcpdump -w)")
    link = struct.unpack(end + "I", data[20:24])[0]
    if link not in LINK_HEADER:
        sys.exit(f"{path}: link type {link} is not read")
    pos = 24
    while pos + 16 <= len(data):
        sec, _, size, _ = struct.unpack(end + "IIII", data[pos:pos + 16])
        frame = data[pos + 16:pos + 16 + size]
        pos += 16 + size
        if link == 1:
            proto, start = struct.unpack(">H", frame[12:14])[0], 14
            if proto == 0x8100:
                proto, start = struct.unpack(">H", frame[16:18])[0], 18
        elif link == 113:
            proto, start = struct.unpack(">H", frame[14:16])[0], 16
        else:
            proto, start = struct.unpack(">H", frame[0:2])[0], 20
        if proto in (0x0800, 0x86DD):
            yield sec, frame[start:]


def parse_ip(pkt):
    """(src, dst, protocol, payload) or None."""
    if not pkt:
        return None
    if pkt[0] >> 4 == 4:
        ihl = (pkt[0] & 15) * 4
        return ipaddress.ip_address(pkt[12:16]), ipaddress.ip_address(pkt[16:20]), pkt[9], pkt[ihl:]
    if pkt[0] >> 4 == 6 and len(pkt) >= 40:
        return ipaddress.ip_address(pkt[8:24]), ipaddress.ip_address(pkt[24:40]), pkt[6], pkt[40:]
    return None


def dns_name(msg, pos):
    """(name, position after the name) with compression pointers."""
    labels, jumped, after = [], False, pos
    for _ in range(128):
        n = msg[pos]
        if n == 0:
            pos += 1
            break
        if n & 0xC0 == 0xC0:
            if not jumped:
                after = pos + 2
            pos, jumped = ((n & 0x3F) << 8) | msg[pos + 1], True
            continue
        labels.append(msg[pos + 1:pos + 1 + n].decode("ascii", "replace"))
        pos += 1 + n
    return ".".join(labels).lower(), (after if jumped else pos)


def dns_answers(msg):
    """[(asked name, address)] from a DNS response."""
    if len(msg) < 12 or not msg[2] & 0x80:
        return []
    qd, an = struct.unpack(">HH", msg[4:8])
    pos, asked = 12, None
    for _ in range(qd):
        asked, pos = dns_name(msg, pos)
        pos += 4
    out = []
    for _ in range(an):
        _, pos = dns_name(msg, pos)
        rtype, _, _, rdlen = struct.unpack(">HHIH", msg[pos:pos + 10])
        rdata = msg[pos + 10:pos + 10 + rdlen]
        pos += 10 + rdlen
        if rtype in (1, 28) and asked:
            out.append((asked, ipaddress.ip_address(rdata)))
    return out


def tls_sni(payload):
    """The server name of a TLS ClientHello, or None."""
    try:
        if payload[0] != 0x16 or payload[5] != 0x01:
            return None
        pos = 9 + 2 + 32                     # record header, handshake header, version, random
        pos += 1 + payload[pos]              # session id
        pos += 2 + struct.unpack(">H", payload[pos:pos + 2])[0]   # cipher suites
        pos += 1 + payload[pos]              # compression methods
        end = pos + 2 + struct.unpack(">H", payload[pos:pos + 2])[0]
        pos += 2
        while pos + 4 <= end:
            etype, elen = struct.unpack(">HH", payload[pos:pos + 4])
            if etype == 0:
                nlen = struct.unpack(">H", payload[pos + 7:pos + 9])[0]
                if pos + 9 + nlen > len(payload):   # the ClientHello goes on in the next TCP segment
                    return None
                return payload[pos + 9:pos + 9 + nlen].decode("ascii", "replace").lower()
            pos += 4 + elen
    except (IndexError, struct.error):
        return None
    return None


def scan(path):
    names = collections.defaultdict(set)     # address -> DNS names
    conns = {}                                # (dst, port, src port) -> first time
    sni = {}                                  # (dst, port, src port) -> server name
    http = []                                 # (time, host, request line)
    for t, pkt in packets(path):
        ip = parse_ip(pkt)
        if not ip:
            continue
        src, dst, proto, body = ip
        if proto == 17 and len(body) >= 8 and struct.unpack(">H", body[0:2])[0] == 53:
            for asked, addr in dns_answers(body[8:]):
                names[addr].add(asked)
        if proto != 6 or len(body) < 20:
            continue
        sport, dport = struct.unpack(">HH", body[0:4])
        flags, payload = body[13], body[(body[12] >> 4) * 4:]
        key = (dst, dport, sport)
        if flags & 0x02 and not flags & 0x10:
            conns.setdefault(key, t)
        name = tls_sni(payload) if payload[:1] == b"\x16" else None
        if name:
            sni[key] = name
        if payload.startswith(b"GET ") and dport == 80:
            head = payload.split(b"\r\n")
            host = next((h[5:].strip().decode() for h in head if h.lower().startswith(b"host:")), str(dst))
            http.append((t, host, head[0].decode("ascii", "replace")))
    return names, conns, sni, http


def when(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S")


def report_network(path):
    names, conns, sni, http = scan(path)
    sni_by_dst = collections.defaultdict(set)
    for (dst, port, _), name in sni.items():
        sni_by_dst[(dst, port)].add(name)
    rows, local = collections.defaultdict(lambda: dict(ports=set(), ips=set(), n=0, first=None, last=None, how="")), set()
    for (dst, port, sport), t in conns.items():
        if dst.is_private or dst.is_loopback or dst.is_link_local:
            local.add(f"{dst}:{port}")
            continue
        if (dst, port, sport) in sni:
            host, how = sni[(dst, port, sport)], "TLS SNI"
        elif sni_by_dst.get((dst, port)):
            host, how = ",".join(sorted(sni_by_dst[(dst, port)])), "TLS SNI"
        elif names.get(dst):
            host, how = ",".join(sorted(names[dst])), "DNS"
        else:
            host, how = str(dst), "address only"
        r = rows[host]
        r["ports"].add(port), r["ips"].add(str(dst))
        r["n"] += 1
        r["first"] = t if r["first"] is None else min(r["first"], t)
        r["last"] = t if r["last"] is None else max(r["last"], t)
        r["how"] = how
    out = ["## Connections", "",
           "Each new outbound TCP connection (SYN), named by TLS SNI, else by the DNS answer for its address. Duplicate "
           "SYNs seen on several interfaces (container, bridge, VM) count once per source port.", "",
           "| Host | Named by | Ports | Addresses | Connections | First (UTC) | Last (UTC) |", "|---|---|---|---|---|---|---|"]
    for host, r in sorted(rows.items(), key=lambda kv: kv[1]["first"]):
        ips = sorted(r["ips"])
        shown = ", ".join(ips[:3]) + (f" (+{len(ips) - 3})" if len(ips) > 3 else "")
        out.append(f"| {host} | {r['how']} | {', '.join(map(str, sorted(r['ports'])))} | {shown} | {r['n']} | "
                   f"{when(r['first'])} | {when(r['last'])} |")
    out += ["", f"Local addresses (VM, bridge, DNS forwarder; not the internet): {', '.join(sorted(local)) or 'none'}", ""]
    out += ["## Plain HTTP requests (port 80)", ""]
    counts = collections.Counter((host, line.split(" ")[1].rsplit("/", 1)[0]) for _, host, line in http)
    out += [f"- {host} {prefix}/... : {n} requests" for (host, prefix), n in sorted(counts.items())] or ["- none"]
    return out


def report_log(path):
    log = open(path, errors="replace").read()
    out = ["", "## What the build fetched (from the build log)", ""]
    images = sorted(set(re.findall(r"(?:FROM|load metadata for) (\S+@sha256:[0-9a-f]{64}|\S+:\S+)", log)))
    out += ["Images:"] + [f"- {i}" for i in images] + [""]
    apt_hosts = collections.Counter(re.findall(r"Get:\d+ (https?://[^/\s]+/\S*)", log))
    debs = re.findall(r"Get:\d+ \S+ \S+ amd64 (\S+) amd64 (\S+)", log)
    out += [f"apt: {sum(apt_hosts.values())} index or package downloads, {len(debs)} packages"]
    out += [f"- {u}: {n}" for u, n in sorted(apt_hosts.items())] + [""]
    pins = re.findall(r"PIN (\S+) ([0-9a-f]{40})", log)
    out += ["git clones (setup.sh PIN lines; submodules are fetched by the same clone):"]
    out += [f"- {name} {sha}" for name, sha in pins] or ["- none"]
    pkgs = sorted(set(re.findall(r"^#\d+ [\d.]+\s+\+ (\S+==\S+|\S+ @ \S+)", log, re.M)))
    out += ["", f"Python packages installed by uv: {len(pkgs)}"]
    out += [f"- {p}" for p in pkgs if " @ " in p]
    out += [f"- {len([p for p in pkgs if '==' in p])} packages pinned with ==version (requirements.txt)"]
    py = sorted(set(re.findall(r"Installed Python (\S+)", log)))
    out += [f"- Python: {', '.join(py) or 'not seen in the log'}"]
    return out


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    lines = ["# Network report of a container build", ""] + report_network(sys.argv[1])
    if len(sys.argv) == 3:
        lines += report_log(sys.argv[2])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
