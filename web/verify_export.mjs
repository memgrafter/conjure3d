// Checks the web export: per object, web.glb and proxy.glb have the master's frame (world bounding box
// vs asset.glb, as a fraction of its diagonal), carry no metadata (generator, copyright, names, extras, URIs, KTX
// writer strings), and lists sizes; prints totals per level.
//   SRC=... OUT=... bun verify_export.mjs [--src asset.glb|raw.glb]  -> $OUT/export-verify.tsv (or $VERIFY_TSV)
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { getBounds } from '@gltf-transform/functions';
import { MeshoptDecoder } from 'meshoptimizer';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const SRC = process.env.SRC, OUT = process.env.OUT;
if ( ! SRC || ! OUT ) { console.error( 'set SRC (folder with <id>/asset.glb) and OUT (export folder)' ); process.exit( 2 ); }
const srcName = process.argv.includes( '--src' ) ? process.argv[ process.argv.indexOf( '--src' ) + 1 ] : 'asset.glb';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions( ALL_EXTENSIONS ).registerDependencies( { 'meshopt.decoder': MeshoptDecoder } );
const LEVELS = [ 'web', 'proxy' ];
const BANNED = /KTXwriter|Blender|glTF-Transform|trimesh|generator|copyright|\/Users\/|\/Volumes\/|\/home\//;

function jsonChunk( buf ) {
	return JSON.parse( buf.subarray( 20, 20 + buf.readUInt32LE( 12 ) ).toString() );
}

function metadataProblems( buf ) {
	const j = jsonChunk( buf ), out = [];
	if ( Object.keys( j.asset ).some( k => k !== 'version' ) ) out.push( 'asset fields' );
	if ( j.extras ) out.push( 'root extras' );
	for ( const key of [ 'scenes', 'nodes', 'meshes', 'materials', 'textures', 'images', 'accessors', 'buffers', 'bufferViews', 'samplers' ] ) {
		for ( const p of j[ key ] || [] ) {
			if ( p.name ) out.push( `${ key } name` );
			if ( p.extras ) out.push( `${ key } extras` );
			if ( p.uri ) out.push( `${ key } uri` );
		}
	}
	const m = buf.toString( 'latin1' ).match( BANNED );
	if ( m ) out.push( `string ${ m[ 0 ] }` );
	return [ ...new Set( out ) ];
}

const rows = [], totals = Object.fromEntries( LEVELS.map( l => [ l, 0 ] ) );
for ( const id of readdirSync( OUT ).filter( id => existsSync( join( OUT, id, 'web.glb' ) ) ).sort() ) {
	const master = getBounds( ( await io.read( join( SRC, id, srcName ) ) ).getRoot().listScenes()[ 0 ] );
	const diag = Math.hypot( ...master.max.map( ( v, i ) => v - master.min[ i ] ) );
	const row = { id, diag_m: +diag.toFixed( 3 ) };
	for ( const level of LEVELS ) {
		const path = join( OUT, id, `${ level }.glb` );
		if ( ! existsSync( path ) ) { row[ `${ level }_bytes` ] = 'missing'; continue; }
		const buf = readFileSync( path );
		const b = getBounds( ( await io.readBinary( new Uint8Array( buf ) ) ).getRoot().listScenes()[ 0 ] );
		const dev = Math.max( ...[ 0, 1, 2 ].flatMap( i => [ Math.abs( b.min[ i ] - master.min[ i ] ), Math.abs( b.max[ i ] - master.max[ i ] ) ] ) ) / diag;
		row[ `${ level }_bytes` ] = statSync( path ).size;
		row[ `${ level }_bbox_dev` ] = +dev.toFixed( 4 );
		row[ `${ level }_metadata` ] = metadataProblems( buf ).join( ';' ) || 'clean';
		totals[ level ] += statSync( path ).size;
	}
	rows.push( row );
}
const cols = [ ...new Set( rows.flatMap( r => Object.keys( r ) ) ) ];
writeFileSync( process.env.VERIFY_TSV || join( OUT, 'export-verify.tsv' ), [ cols.join( '\t' ), ...rows.map( r => cols.map( c => r[ c ] ?? '' ).join( '\t' ) ) ].join( '\n' ) + '\n' );
const worst = l => rows.reduce( ( a, r ) => ( r[ `${ l }_bbox_dev` ] ?? 0 ) > ( a[ `${ l }_bbox_dev` ] ?? 0 ) ? r : a, {} );
for ( const l of LEVELS ) {
	const sizes = rows.map( r => r[ `${ l }_bytes` ] ).filter( Number.isFinite ).sort( ( a, b ) => a - b );
	const dirty = rows.filter( r => r[ `${ l }_metadata` ] && r[ `${ l }_metadata` ] !== 'clean' );
	console.log( `${ l }: ${ sizes.length } files, total ${ totals[ l ] } bytes (${ ( totals[ l ] / 1e6 ).toFixed( 2 ) } MB), min ${ sizes[ 0 ] }, median ${ sizes[ sizes.length >> 1 ] }, max ${ sizes.at( - 1 ) }; worst bbox deviation ${ worst( l )[ `${ l }_bbox_dev` ] } (${ worst( l ).id }); metadata problems: ${ dirty.length ? dirty.map( r => `${ r.id }:${ r[ `${ l }_metadata` ] }` ).join( ' ' ) : 'none' }` );
}
