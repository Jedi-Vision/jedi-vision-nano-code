#!/bin/bash

SRC_DIR="."
DST_DIR="../jv/pb"

protoc --proto_path=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/objectrep.proto
pbtools generate_c_source objectrep.proto -o $DST_DIR