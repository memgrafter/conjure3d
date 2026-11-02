"""tools/netscan_report.py: a small pcap made here, with one DNS answer, one TLS connection and one HTTP request."""
import struct
from conftest import load

netscan = load("tools/netscan_report.py", "netscan_report")
VM, SERVER, DNS = bytes([172, 17, 0, 2]), bytes([93, 184, 216, 34]), bytes([192, 168, 5, 1])


def ipv4(src, dst, proto, body):
    return struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(body), 0, 0, 64, proto, 0, src, dst) + body


def tcp(sport, dport, flags, payload=b""):
    return struct.pack(">HHIIBBHHH", sport, dport, 0, 0, 5 << 4, flags, 0, 0, 0) + payload


def name(text):
    return b"".join(bytes([len(p)]) + p.encode() for p in text.split(".")) + b"\0"


def dns_response(asked, addr):
    question = name(asked) + struct.pack(">HH", 1, 1)
    answer = b"\xc0\x0c" + struct.pack(">HHIH", 1, 1, 60, 4) + addr
    msg = struct.pack(">HHHHHH", 1, 0x8180, 1, 1, 0, 0) + question + answer
    return struct.pack(">HHHH", 53, 40000, 8 + len(msg), 0) + msg


def client_hello(host):
    sni = struct.pack(">HB", len(host) + 3, 0) + struct.pack(">H", len(host)) + host.encode()
    ext = struct.pack(">HH", 0, len(sni)) + sni
    body = b"\x03\x03" + bytes(32) + b"\0" + struct.pack(">H", 2) + b"\x13\x01" + b"\x01\0" + struct.pack(">H", len(ext)) + ext
    hs = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack(">H", len(hs)) + hs


def sll2(ip):
    return struct.pack(">HHIHBB8s", 0x0800, 0, 1, 0, 4, 6, bytes(8)) + ip


def write_pcap(path, frames):
    data = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 276)
    for t, frame in enumerate(frames):
        data += struct.pack("<IIII", 1_700_000_000 + t, 0, len(frame), len(frame)) + frame
    open(path, "wb").write(data)


def test_report_names_connections(tmp_path):
    pcap = str(tmp_path / "b.pcap")
    write_pcap(pcap, [
        sll2(ipv4(DNS, VM, 17, dns_response("example.org", SERVER))),
        sll2(ipv4(VM, SERVER, 6, tcp(50000, 443, 0x02))),
        sll2(ipv4(VM, SERVER, 6, tcp(50000, 443, 0x02))),            # the same SYN seen on a second interface
        sll2(ipv4(VM, SERVER, 6, tcp(50000, 443, 0x18, client_hello("files.example.org")))),
        sll2(ipv4(VM, SERVER, 6, tcp(50001, 80, 0x02))),
        sll2(ipv4(VM, SERVER, 6, tcp(50001, 80, 0x18, b"GET /ubuntu/pool/a.deb HTTP/1.1\r\nHost: archive.example\r\n\r\n"))),
    ])
    text = "\n".join(netscan.report_network(pcap))
    assert "| files.example.org | TLS SNI | 443 | 93.184.216.34 | 1 |" in text
    assert "| example.org | DNS | 80 | 93.184.216.34 | 1 |" in text
    assert "archive.example /ubuntu/pool/... : 1 requests" in text


def test_sni_cut_at_the_segment_end_is_not_used():
    hello = client_hello("download.pytorch.org")
    assert netscan.tls_sni(hello) == "download.pytorch.org"
    assert netscan.tls_sni(hello[:-4]) is None
