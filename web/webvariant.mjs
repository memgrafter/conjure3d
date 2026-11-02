// Builds web variants of one TRELLIS.2 GLB: fewer triangles, smaller textures,
// WebP or KTX2 (Basis ETC1S / UASTC) textures, meshopt geometry. One process per source file; textures are encoded
// once per (slot, size, format) and shared by the variants.
//   bun webvariant.mjs SRC OUT_DIR [--render] NAME=TRIS:TEX:FMT[:MR[:FLAGS]] [NAME=...]
//     TRIS  target triangle count, or "full"
//     TEX   longest side of the base colour texture (px)
//     FMT   webp | etc1s | uastc  (base colour and metallic-roughness)
//     MR    metallic-roughness size in px (default TEX/2, at most 1024), or "none": drop it and use its mean as factors
//     FLAGS comma list: loose = simplify with no error bound, Permissive (UV seams collapse; for meshes whose textures
//           are re-baked on new UVs, bake_proxy.py); lowq = coarser quantization for tiny proxies (OPT.lowQuantize);
//           fill = refill the atlas between UV charts (fillUnused). Off by default: measured worse than TRELLIS's own
//           padding in a size/quality sweep, kept for the record
//   -> OUT_DIR/NAME.glb and OUT_DIR/NAME.json (sizes, triangles, simplify error, GPU memory)
//   --render also writes OUT_DIR/NAME.render.glb: the same geometry and decoded texture pixels with no compression
//   extensions, for Blender (KTX2 decoded with basisu to the pixels a browser gets; Blender cannot read KTX2).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTTextureWebP, KHRTextureBasisu } from '@gltf-transform/extensions';
import { cloneDocument, compactPrimitive, dedup, dequantize, meshopt, prune, weld } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';
import { read as readKtx, write as writeKtx } from 'ktx-parse';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

const BASISU = process.env.BASISU || 'basisu';
// Pinned options (chosen in a size/quality sweep of TRELLIS.2 assets).
const OPT = {
	simplifyError: 0.05, // upper bound on the simplifier's error, relative to the mesh extent
	simplifyFlags: [ 'LockBorder' ], // open borders stay; UV/normal seams are kept by meshoptimizer's default seam rules
	webpQuality: 80,
	etc1sQ: 255,
	uastcLevel: 2,
	uastcRdoLambda: 1,
	zstdLevel: 18,
	mrMax: 1024,
	// meshopt 'high' quantizes positions to 14 bits, texcoords to 12, normals to 8; proxies (128 px texture, seen far
	// away) keep 11-bit positions (1/2048 of the extent) and 10-bit texcoords
	lowQuantize: { quantizePosition: 11, quantizeTexcoord: 10 },
};

const [ src, outDir, ...rest ] = process.argv.slice( 2 );
const wantRender = rest.includes( '--render' );
const specs = rest.filter( a => a.includes( '=' ) ).map( parseSpec );
if ( ! src || ! outDir || specs.length === 0 ) {
	console.error( 'usage: bun webvariant.mjs SRC OUT_DIR [--render] NAME=TRIS:TEX:FMT[:MR] ...' );
	process.exit( 1 );
}
mkdirSync( outDir, { recursive: true } );
const work = join( tmpdir(), 'webvariant-work', String( process.pid ) );
for ( const d of [ 'ETC2_RGBA', 'ASTC_LDR_4X4_RGBA' ] ) mkdirSync( join( work, d ), { recursive: true } );

function parseSpec( arg ) {
	const [ name, body ] = arg.split( '=' );
	const [ tris, tex, fmt, mr, flagList = '' ] = body.split( ':' );
	const flags = flagList.split( ',' );
	if ( ! /^[a-z0-9_.-]+$/i.test( name ) || ! [ 'webp', 'etc1s', 'uastc' ].includes( fmt ) ) throw new Error( `bad spec ${ arg }` );
	const texSize = Number( tex );
	const mrSize = mr === 'none' ? 0 : mr ? Number( mr ) : Math.min( OPT.mrMax, texSize / 2 );
	return { name, tris: tris === 'full' ? Infinity : Number( tris ), texSize, fmt, mrSize, loose: flags.includes( 'loose' ), fill: flags.includes( 'fill' ), lowq: flags.includes( 'lowq' ) };
}

await Promise.all( [ MeshoptEncoder.ready, MeshoptDecoder.ready, MeshoptSimplifier.ready ] );
const io = new NodeIO().registerExtensions( ALL_EXTENSIONS )
	.registerDependencies( { 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder } );

const tLoad = performance.now();
const base = await io.read( src );
await base.transform( dedup(), weld() );
const baseFaces = countFaces( base );
const loadSeconds = ( performance.now() - tLoad ) / 1000;

const texCache = new Map();
const filled = new Map(); // source image byte length -> { raw, w, h, ch } with the space between charts filled
for ( const spec of specs ) {
	const t0 = performance.now();
	const doc = cloneDocument( base );
	const simp = spec.tris < baseFaces ? simplifyDocument( doc, spec.tris, spec.loose ) : { faces: baseFaces, error: 0 };
	await doc.transform( prune( { keepSolidTextures: true } ) );
	const tex = await applyTextures( doc, spec );
	await doc.transform( meshopt( { encoder: MeshoptEncoder, level: 'high', ...( spec.lowq ? OPT.lowQuantize : {} ) } ) );
	stripMetadata( doc );
	const bytes = await io.writeBinary( doc );
	writeFileSync( join( outDir, `${ spec.name }.glb` ), bytes );
	const back = await io.readBinary( bytes );
	const stats = {
		name: spec.name, src, tris_target: Number.isFinite( spec.tris ) ? spec.tris : 'full', faces_src: baseFaces,
		faces: simp.faces, loose: spec.loose, lowq: spec.lowq, fill: spec.fill, simplify_error: +simp.error.toFixed( 5 ), vertices: countVertices( back ),
		fmt: spec.fmt, textures: tex, bytes: bytes.byteLength, gzip9_bytes: Bun.gzipSync( bytes, { level: 9 } ).byteLength,
		gpu_geometry_bytes: geometryBytes( back ),
		gpu_texture_bytes_apple: tex.reduce( ( s, t ) => s + t.gpu_apple, 0 ),
		gpu_texture_bytes_desktop: tex.reduce( ( s, t ) => s + t.gpu_desktop, 0 ),
		build_seconds: +( ( performance.now() - t0 ) / 1000 ).toFixed( 2 ), load_seconds: +loadSeconds.toFixed( 2 ),
		options: OPT,
	};
	if ( wantRender ) writeFileSync( join( outDir, `${ spec.name }.render.glb` ), await renderCopy( back ) );
	writeFileSync( join( outDir, `${ spec.name }.json` ), JSON.stringify( stats, null, 1 ) );
	console.log( JSON.stringify( { name: stats.name, faces: stats.faces, err: stats.simplify_error, bytes: stats.bytes, s: stats.build_seconds } ) );
}

// Published files carry no generator, copyright, names, extras or URIs (no personal data, no tool paths).
function stripMetadata( doc ) {
	const root = doc.getRoot();
	const asset = root.getAsset();
	asset.generator = undefined;
	asset.copyright = undefined;
	asset.extras = undefined;
	root.setExtras( {} );
	for ( const list of [ root.listScenes(), root.listNodes(), root.listMeshes(), root.listMaterials(), root.listTextures(), root.listAccessors(), root.listBuffers(), root.listSkins(), root.listAnimations(), root.listCameras() ] ) {
		for ( const p of list ) p.setName( '' ).setExtras( {} );
	}
	for ( const mesh of root.listMeshes() ) for ( const prim of mesh.listPrimitives() ) prim.setExtras( {} );
	for ( const t of root.listTextures() ) t.setURI( '' );
	for ( const b of root.listBuffers() ) b.setURI( '' );
}

function countFaces( doc ) {
	let n = 0;
	for ( const mesh of doc.getRoot().listMeshes() ) for ( const p of mesh.listPrimitives() ) n += ( p.getIndices()?.getCount() ?? p.getAttribute( 'POSITION' ).getCount() ) / 3;
	return n;
}

function countVertices( doc ) {
	let n = 0;
	for ( const mesh of doc.getRoot().listMeshes() ) for ( const p of mesh.listPrimitives() ) n += p.getAttribute( 'POSITION' ).getCount();
	return n;
}

// Bytes the vertex and index buffers take once uploaded: each element padded to 4 bytes (vertex attribute alignment).
function geometryBytes( doc ) {
	const seen = new Set();
	let n = 0;
	for ( const mesh of doc.getRoot().listMeshes() ) for ( const p of mesh.listPrimitives() ) {
		for ( const a of [ ...p.listAttributes(), p.getIndices() ].filter( Boolean ) ) {
			if ( seen.has( a ) ) continue;
			seen.add( a );
			const elem = a.getElementSize() * a.getComponentSize();
			n += a.getCount() * ( a === p.getIndices() ? elem : Math.ceil( elem / 4 ) * 4 );
		}
	}
	return n;
}

function simplifyDocument( doc, targetFaces, loose ) {
	const ratio = targetFaces / countFaces( doc );
	let error = 0;
	for ( const mesh of doc.getRoot().listMeshes() ) for ( const prim of mesh.listPrimitives() ) {
		const srcIndices = prim.getIndices();
		const indices = new Uint32Array( srcIndices.getArray() );
		const position = prim.getAttribute( 'POSITION' ).getArray();
		const target = Math.floor( ratio * indices.length / 3 ) * 3;
		const [ out, err ] = MeshoptSimplifier.simplify( indices, position, 3, target, loose ? 1 : OPT.simplifyError, loose ? [ 'Permissive' ] : OPT.simplifyFlags );
		prim.setIndices( doc.createAccessor().setType( 'SCALAR' ).setArray( out ).setBuffer( srcIndices.getBuffer() ) );
		if ( srcIndices.listParents().length === 1 ) srcIndices.dispose();
		compactPrimitive( prim );
		error = Math.max( error, err );
	}
	return { faces: countFaces( doc ), error };
}

async function applyTextures( doc, spec ) {
	const out = [];
	const root = doc.getRoot();
	for ( const material of root.listMaterials() ) {
		const opaque = material.getAlphaMode() === 'OPAQUE';
		const base = material.getBaseColorTexture();
		const mr = material.getMetallicRoughnessTexture();
		if ( base ) out.push( await setTexture( base, 'base', spec.texSize, spec.fmt, false, opaque, spec.fill ) );
		if ( mr && spec.mrSize === 0 ) {
			// proxy: replace the texture by its mean (glTF: roughness in G, metallic in B)
			const st = await sharp( Buffer.from( mr.getImage() ), { limitInputPixels: false } ).stats();
			material.setRoughnessFactor( material.getRoughnessFactor() * st.channels[ 1 ].mean / 255 );
			material.setMetallicFactor( material.getMetallicFactor() * st.channels[ 2 ].mean / 255 );
			material.setMetallicRoughnessTexture( null );
		} else if ( mr ) out.push( await setTexture( mr, 'mr', spec.mrSize, spec.fmt, true, true, spec.fill ) );
	}
	await doc.transform( prune( { keepSolidTextures: true } ) );
	if ( root.listTextures().length ) doc.createExtension( spec.fmt === 'webp' ? EXTTextureWebP : KHRTextureBasisu ).setRequired( true );
	return out;
}

// Alpha is dropped when the material is OPAQUE: renderers ignore it there, and it would double the KTX2 cost.
async function setTexture( texture, slot, size, fmt, linear, dropAlpha, fill ) {
	const len = texture.getImage().byteLength;
	const key = `${ slot }:${ size }:${ fmt }:${ dropAlpha }:${ fill }:${ len }`;
	if ( fill && ! filled.has( len ) ) filled.set( len, await fillUnused( texture.getImage() ) );
	if ( ! texCache.has( key ) ) texCache.set( key, await encodeTexture( fill ? filled.get( len ) : texture.getImage(), size, fmt, linear, dropAlpha ) );
	const enc = texCache.get( key );
	texture.setImage( enc.image ).setMimeType( enc.mime ).setURI( '' );
	return { slot, w: enc.w, h: enc.h, fmt, bytes: enc.image.byteLength, gpu_apple: enc.gpuApple, gpu_desktop: enc.gpuDesktop, encode_seconds: enc.seconds };
}

async function encodeTexture( image, size, fmt, linear, dropAlpha ) {
	const t0 = performance.now();
	const input = image.raw ? sharp( image.raw, { raw: { width: image.w, height: image.h, channels: image.ch }, limitInputPixels: false } )
		: sharp( Buffer.from( image ), { limitInputPixels: false } );
	// alpha must go before the resize: sharp resizes premultiplied, so RGB under alpha 0 would turn black (an OPAQUE
	// material still shows that RGB; TRELLIS atlases have alpha 0 on visible charts, e.g. the pocket watch's white dial)
	let src = input;
	if ( dropAlpha ) {
		const flat = await input.removeAlpha().raw().toBuffer( { resolveWithObject: true } );
		src = sharp( flat.data, { raw: { width: flat.info.width, height: flat.info.height, channels: flat.info.channels }, limitInputPixels: false } );
	}
	let img = src.resize( size, size, { fit: 'inside', kernel: 'lanczos3', withoutEnlargement: true } );
	const png = await img.png( { compressionLevel: 1 } ).toBuffer( { resolveWithObject: true } );
	const { width: w, height: h, channels } = png.info;
	const hasAlpha = channels === 4;
	let bytes, mime;
	if ( fmt === 'webp' ) {
		bytes = await sharp( png.data ).webp( { quality: OPT.webpQuality } ).toBuffer();
		mime = 'image/webp';
	} else {
		const inPath = join( work, 'in.png' ), outPath = join( work, 'out.ktx2' );
		writeFileSync( inPath, png.data );
		const codec = fmt === 'etc1s' ? [ '-etc1s', '-q', String( OPT.etc1sQ ) ]
			: [ '-uastc', '-uastc_level', String( OPT.uastcLevel ), '-uastc_rdo_l', String( OPT.uastcRdoLambda ), '-ktx2_zstandard_level', String( OPT.zstdLevel ) ];
		const args = [ ...codec, '-mipmap', linear ? '-linear' : '-srgb', '-ktx2', '-file', inPath, '-output_file', outPath ];
		const r = spawnSync( BASISU, args, { encoding: 'utf8', timeout: 600000 } );
		if ( r.status !== 0 ) throw new Error( `basisu failed: ${ r.stderr || r.stdout }` );
		const ktx = readKtx( readFileSync( outPath ) );
		// no tool name/version in published files. The key-value block must not be empty (basisu's transcoder rejects
		// that file), so the writer entry is replaced by the orientation glTF already implies (KTX2 default, right-down)
		delete ktx.keyValue.KTXwriter;
		ktx.keyValue.KTXorientation = 'rd';
		bytes = writeKtx( ktx, { keepWriter: true } );
		mime = 'image/ktx2';
	}
	return {
		image: new Uint8Array( bytes ), mime, w, h,
		gpuApple: gpuTextureBytes( w, h, fmt === 'webp' ? 4 : fmt === 'etc1s' && ! hasAlpha ? 0.5 : 1 ),
		gpuDesktop: gpuTextureBytes( w, h, fmt === 'webp' ? 4 : 1 ),
		seconds: +( ( performance.now() - t0 ) / 1000 ).toFixed( 2 ),
	};
}

// Full mip chain. WebP -> RGBA8 (4 B/px). KTX2 as three.js r187 KTX2Loader transcodes it: Apple/mobile WebGPU picks
// ETC2 RGB for opaque ETC1S (0.5 B/px) and ASTC 4x4 for UASTC (1 B/px); desktop without ETC2/ASTC picks BC7 (1 B/px).
function gpuTextureBytes( w, h, bytesPerPixel ) {
	let n = 0;
	for ( ; ; w = Math.max( 1, w >> 1 ), h = Math.max( 1, h >> 1 ) ) {
		n += bytesPerPixel === 4 ? w * h * 4 : Math.ceil( w / 4 ) * Math.ceil( h / 4 ) * 16 * bytesPerPixel;
		if ( w === 1 && h === 1 ) break;
	}
	return n;
}

// Decoded copy for Blender: meshopt removed, attributes dequantized, textures as PNG with the pixels a browser gets
// (KTX2: basisu's transcode to ETC2 for ETC1S, ASTC 4x4 for UASTC, the Apple WebGPU targets).
async function renderCopy( doc ) {
	await doc.transform( dequantize() );
	for ( const ext of doc.getRoot().listExtensionsUsed() ) {
		if ( [ 'EXT_meshopt_compression', 'KHR_mesh_quantization', 'EXT_texture_webp', 'KHR_texture_basisu' ].includes( ext.extensionName ) ) ext.dispose();
	}
	for ( const t of doc.getRoot().listTextures() ) {
		const mime = t.getMimeType();
		if ( mime === 'image/webp' ) t.setImage( new Uint8Array( await sharp( Buffer.from( t.getImage() ) ).png().toBuffer() ) );
		else if ( mime === 'image/ktx2' ) t.setImage( decodeKtx2( t.getImage() ) );
		t.setMimeType( 'image/png' );
	}
	return io.writeBinary( doc );
}

function decodeKtx2( image ) {
	const scheme = new DataView( image.buffer, image.byteOffset ).getUint32( 44, true ); // 1 BasisLZ (ETC1S), 2 Zstd (UASTC)
	const [ format, label ] = scheme === 1 ? [ '1', 'ETC2_RGBA' ] : [ '10', 'ASTC_LDR_4X4_RGBA' ];
	const dir = join( work, label ); // one folder per format: the fixed file names are overwritten, never mixed
	writeFileSync( join( dir, 'dec.ktx2' ), image );
	const r = spawnSync( BASISU, [ '-unpack', '-no_ktx', '-format_only', format, 'dec.ktx2' ], { cwd: dir, encoding: 'utf8', timeout: 600000 } );
	if ( r.status !== 0 ) throw new Error( `basisu unpack failed: ${ r.stderr || r.stdout }` );
	const name = readdirSync( dir ).find( f => f === `dec_unpacked_rgba_${ label }_level_0_face_0_layer_0000.png` );
	if ( ! name ) throw new Error( 'basisu unpack: no level 0 RGBA png' );
	return new Uint8Array( readFileSync( join( dir, name ) ) );
}

// TRELLIS atlases put unrelated colour (often saturated green) in the space between UV charts. Downscaling and
// mipmapping blend it into the chart edges, which shows as coloured speckles along every seam. fillUnused rasterizes
// the source mesh's UV triangles (texel centres inside a triangle, plus each vertex's texel) and fills every other
// texel by push-pull: average the covered texels down a 2x2 pyramid, then copy each empty texel from its parent.
// Covered texels are unchanged; only chart colours can bleed. All primitives' TEXCOORD_0 are used.
async function fillUnused( image ) {
	const { data, info } = await sharp( Buffer.from( image ), { limitInputPixels: false } ).raw().toBuffer( { resolveWithObject: true } );
	const { width: w, height: h, channels: ch } = info;
	const mask = new Uint8Array( w * h );
	for ( const mesh of base.getRoot().listMeshes() ) for ( const prim of mesh.listPrimitives() ) {
		const uv = prim.getAttribute( 'TEXCOORD_0' );
		if ( ! uv ) continue;
		const t = uv.getArray(), idx = prim.getIndices().getArray();
		for ( let i = 0; i < idx.length; i += 3 ) rasterize( mask, w, h, t, idx[ i ], idx[ i + 1 ], idx[ i + 2 ] );
	}
	const levels = [ { w, h, col: new Float32Array( w * h * ch ), wt: new Float32Array( w * h ) } ];
	for ( let p = 0; p < w * h; p ++ ) if ( mask[ p ] ) {
		levels[ 0 ].wt[ p ] = 1;
		for ( let c = 0; c < ch; c ++ ) levels[ 0 ].col[ p * ch + c ] = data[ p * ch + c ];
	}
	while ( levels.at( - 1 ).w > 1 || levels.at( - 1 ).h > 1 ) {
		const a = levels.at( - 1 ), nw = Math.max( 1, a.w >> 1 ), nh = Math.max( 1, a.h >> 1 );
		const b = { w: nw, h: nh, col: new Float32Array( nw * nh * ch ), wt: new Float32Array( nw * nh ) };
		for ( let y = 0; y < a.h; y ++ ) for ( let x = 0; x < a.w; x ++ ) {
			const p = y * a.w + x, q = Math.min( y >> 1, nh - 1 ) * nw + Math.min( x >> 1, nw - 1 );
			b.wt[ q ] += a.wt[ p ];
			for ( let c = 0; c < ch; c ++ ) b.col[ q * ch + c ] += a.col[ p * ch + c ];
		}
		levels.push( b );
	}
	for ( const l of levels ) for ( let p = 0; p < l.w * l.h; p ++ ) if ( l.wt[ p ] > 0 ) for ( let c = 0; c < ch; c ++ ) l.col[ p * ch + c ] /= l.wt[ p ];
	for ( let i = levels.length - 2; i >= 0; i -- ) {
		const a = levels[ i ], b = levels[ i + 1 ];
		for ( let y = 0; y < a.h; y ++ ) for ( let x = 0; x < a.w; x ++ ) {
			const p = y * a.w + x;
			if ( a.wt[ p ] > 0 ) continue;
			const q = Math.min( y >> 1, b.h - 1 ) * b.w + Math.min( x >> 1, b.w - 1 );
			for ( let c = 0; c < ch; c ++ ) a.col[ p * ch + c ] = b.col[ q * ch + c ];
			a.wt[ p ] = 1;
		}
	}
	const raw = Buffer.alloc( w * h * ch );
	for ( let p = 0; p < w * h * ch; p ++ ) raw[ p ] = mask[ ( p / ch ) | 0 ] ? data[ p ] : Math.round( levels[ 0 ].col[ p ] );
	let covered = 0;
	for ( let p = 0; p < w * h; p ++ ) covered += mask[ p ];
	console.error( `fill: ${ w }x${ h }, ${ ( 100 * covered / ( w * h ) ).toFixed( 1 ) }% of texels covered by UV triangles` );
	return { raw, w, h, ch };
}

function rasterize( mask, w, h, t, i0, i1, i2 ) {
	const ax = t[ 2 * i0 ] * w, ay = t[ 2 * i0 + 1 ] * h, bx = t[ 2 * i1 ] * w, by = t[ 2 * i1 + 1 ] * h, cx = t[ 2 * i2 ] * w, cy = t[ 2 * i2 + 1 ] * h;
	for ( const [ px, py ] of [ [ ax, ay ], [ bx, by ], [ cx, cy ] ] ) {
		const x = Math.min( w - 1, Math.max( 0, Math.floor( px ) ) ), y = Math.min( h - 1, Math.max( 0, Math.floor( py ) ) );
		mask[ y * w + x ] = 1;
	}
	const area = ( bx - ax ) * ( cy - ay ) - ( by - ay ) * ( cx - ax );
	if ( area === 0 ) return;
	const s = area > 0 ? 1 : - 1;
	const x0 = Math.max( 0, Math.ceil( Math.min( ax, bx, cx ) - 0.5 ) ), x1 = Math.min( w - 1, Math.floor( Math.max( ax, bx, cx ) - 0.5 ) );
	const y0 = Math.max( 0, Math.ceil( Math.min( ay, by, cy ) - 0.5 ) ), y1 = Math.min( h - 1, Math.floor( Math.max( ay, by, cy ) - 0.5 ) );
	for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) {
		const px = x + 0.5, py = y + 0.5;
		const e0 = s * ( ( bx - ax ) * ( py - ay ) - ( by - ay ) * ( px - ax ) );
		const e1 = s * ( ( cx - bx ) * ( py - by ) - ( cy - by ) * ( px - bx ) );
		const e2 = s * ( ( ax - cx ) * ( py - cy ) - ( ay - cy ) * ( px - cx ) );
		if ( e0 >= 0 && e1 >= 0 && e2 >= 0 ) mask[ y * w + x ] = 1;
	}
}
