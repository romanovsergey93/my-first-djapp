from neo import io
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="link to the ABF file")
parser.add_argument('--link', help='link to the ABF file', type=str, default='F:\\abfs\\17208000.abf')

def impt():

	args = parser.parse_args()
	link = args.link
	print("link")

	read_abf = io.AxonIO(filename=link)
	read_blocks = read_abf.read_block(lazy=False, cascade=True)
	reader_data = read_blocks.segments[0].analogsignals

	print("INFO #1")
	print(read_abf.name, read_abf.description, read_abf.extensions, \
	        read_abf.extentions, read_abf.filename, read_abf.has_header, \
	        read_abf.is_readable, read_abf.is_streameable, \
	        read_abf.is_writable, read_abf.logger, read_abf.mode)

	print("INFO #2")
	print(read_blocks.annotations, read_abf.description, \
	        read_blocks.file_datetime, read_blocks.file_origin, \
	        read_blocks.index, read_blocks.name, read_blocks.rec_datetime)

	print("INFO #3")
	print(read_blocks, read_blocks.segments, read_blocks.segments, \
	        read_blocks.segments[0].analogsignals)

	current = np.array(reader_data[0], float)
	length = np.array(reader_data[1], float)
	voltage = np.array(reader_data[2], float)

	current = np.reshape(current, len(current))
	length = np.reshape(length, len(length))
	voltage = np.reshape(voltage, len(voltage))

	return current, length, voltage

if __name__ == "__impt__":
	sys.exit(impt())

print(main())
