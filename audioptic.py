import subprocess
import cv2 as cv
import imageio_ffmpeg as ii
import numpy as np
ffmpeg = ii.get_ffmpeg_exe()
ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-v", "info", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto")
C0 = 440 / 32 * 2 ** (3 / 12)

IMAGE_FORMS = (
	"gif",
	"png",
	"apng",
	"bmp",
	"jpg",
	"jpeg",
	"jp2",
	"jpx",
	"jxl",
	"tiff",
	"webp",
	"heic",
	"heif",
	"avif",
	"ico",
)
def is_url(url):
	return "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
def bps_format(fmt):
	if fmt in ("wav", "flac"):
		return "1536k"
	if fmt == "mp3":
		return "224k"
	if fmt in ("aac", "m4a"):
		return "192k"
	return "160k"

FFTS = 3176
DFTS = FFTS // 2 + 1
BASE = 8
MAX = 24

def transform(data):
	transformed = []
	for channel in data:
		temp = []
		for i in range(len(channel) // FFTS):
			temp.append(np.fft.rfft(channel[i * FFTS:i * FFTS + FFTS])[::-1][1:].astype(np.complex64))
		transformed.append(temp)
	return np.array(transformed).swapaxes(1, 2)
def itransform(transformed):
	data = []
	for channel in transformed:
		temp = []
		for block in channel.T:
			temp.append(np.fft.irfft(np.pad(block, (1, 0), mode="edge")[::-1]).astype(np.float32))
		data.append(np.concatenate(temp))
	return np.array(data)

def audio2image(ctx):
	args = ffmpeg_start + ("-vn", "-i", ctx.input, "-f", "f32le", "-ac", "2", "-ar", str(ctx.sample_rate), "-")
	stdout = subprocess.check_output(args)
	data = np.frombuffer(stdout, dtype=np.float32)
	data = data.reshape((len(data) // 2), 2).T
	transformed = transform(data)

	ima = np.empty((transformed.shape[1], transformed.shape[2] * 2, 3), dtype=np.float32)
	left_amp, right_amp = np.abs(transformed)
	left_pha, right_pha = np.angle(transformed)
	ima[:, ::2, 0] = left_pha * (180 / np.pi) + 180
	ima[:, 1::2, 0] = right_pha * (180 / np.pi) + 180
	left_exp = np.clip((np.ceil(np.log2(left_amp)) + BASE) * (1 / MAX), 0, 1)
	right_exp = np.clip((np.ceil(np.log2(right_amp)) + BASE) * (1 / MAX), 0, 1)
	left_frac = left_amp / 2 ** (left_exp * MAX - BASE)
	right_frac = right_amp / 2 ** (right_exp * MAX - BASE)
	ima[:, ::2, 1] = left_exp
	ima[:, 1::2, 1] = right_exp
	ima[:, ::2, 2] = left_frac
	ima[:, 1::2, 2] = right_frac
	im = np.round(cv.cvtColor(ima, cv.COLOR_HLS2BGR) * 255).astype(np.uint8)
	if ctx.format in ("png", "webp", "tiff", "jpg", "bmp"):
		cv.imwrite(ctx.output, im)
	else:
		args = ffmpeg_start + ("-f", "rawvideo", "-video_size", f"{im.shape[1]}x{im.shape[0]}", "-pix_fmt", "bgr24", "-r", "1", "-i", "-", "-vframes", "1", "-b:v", "9999999", "-quality", "100", "-lossless", "1", ctx.output)
		print(args)
		subprocess.Popen(args, stdin=subprocess.PIPE).communicate(im.data)

def image2audio(ctx):
	input2 = None
	if ctx.format not in ("png", "webp", "tiff", "jpg", "bmp"):
		input2 = ctx.input + ".png"
		args = ffmpeg_start + ("-i", ctx.input, "-vframes", "1", "-lossless", "1", input2)
		print(args)
		subprocess.run(args)
		ctx.input = input2
	try:
		im = cv.imread(ctx.input)
	finally:
		if input2:
			import os
			os.remove(input2)
	assert im is not None, "Image was not loaded properly!"
	ima = cv.cvtColor(im.astype(np.float32) * (1 / 255), cv.COLOR_BGR2HLS)
	left_pha = (ima[:, ::2, 0] - 180) * (np.pi / 180)
	right_pha = (ima[:, 1::2, 0] - 180) * (np.pi / 180)
	left_amp = ima[:, ::2, 2] * 2 ** (np.round(ima[:, ::2, 1] * MAX) - BASE)
	right_amp = ima[:, 1::2, 2] * 2 ** (np.round(ima[:, 1::2, 1] * MAX) - BASE)
	left_amp[ima[:, ::2, 1] == 0] = 0
	right_amp[ima[:, 1::2, 1] == 0] = 0
	left = left_amp * np.exp(1j * left_pha, dtype=np.complex64)
	right = right_amp * np.exp(1j * right_pha, dtype=np.complex64)
	transformed = np.stack([left[:, :right.shape[1]], right])

	data = np.clip(itransform(transformed), -2, 2)
	codec = ["-sample_fmt", "s16"] if ctx.format in ("wav", "flac") else ["-c:a", "libopus", "-vbr", "on"] if ctx.format in ("ogg", "opus") else ["-vbr", "on"]
	args = ffmpeg_start + ("-f", "f32le", "-ac", "2", "-ar", str(ctx.sample_rate), "-i", "-", "-f", ctx.format, *codec, "-b:a", bps_format(ctx.format), ctx.output)
	print(args)
	subprocess.Popen(args, stdin=subprocess.PIPE).communicate(data.T.tobytes())


try:
	from importlib.metadata import version
	__version__ = version("audioptic")
except Exception:
	__version__ = "0.0.0-unknown"

def main():
	import argparse
	import filetype
	parser = argparse.ArgumentParser(
		prog="audioptic",
		description="Bidirectional spectrogram-based audio-image converter",
	)
	parser.add_argument("-V", '--version', action='version', version=f'%(prog)s {__version__}')
	parser.add_argument("input", help="Input filename or URL")
	parser.add_argument("-sr", "--sample_rate", help="Sample rate; defaults to 42000", nargs="?", type=int, default=42000)
	parser.add_argument("-f", "--format", help="Output format; defaults to opus or webp depending on input", nargs="?", default="")
	parser.add_argument("output", help="Output filename", nargs="?", default="")
	args = parser.parse_args()
	input2 = None
	if is_url(args.input):
		input2 = args.input.split("?", 1)[0].rsplit("/", 1)[-1]
		if "." in input2:
			name, ext = input2.rsplit(".", 1)
			input2 = name + "~." + ext
		else:
			input2 += "~"
		import asyncio
		import streamshatter
		asyncio.run(streamshatter.shatter_request(args.input, filename=input2))
		args.input = input2
	try:
		name = args.input.rsplit(".", 1)[0]
		try:
			ext = filetype.guess(args.input).extension
		except AttributeError:
			ext = args.input.rsplit(".", 1)[-1]
		fmt = args.format
		if not fmt:
			fmt = args.format = args.output.rsplit(".", 1)[-1] if "." in args.output else ("opus" if ext in IMAGE_FORMS else "webp")
		if not args.output:
			args.output = f"{name}.{fmt}"
		if fmt in IMAGE_FORMS:
			if ext in IMAGE_FORMS:
				args = ffmpeg_start + ("-i", args.input, "-vframes", "1", "-b:v", "9999999", "-quality", "100", "-lossless", "1", args.output)
				print(args)
				subprocess.run(args)
			else:
				audio2image(args)
		else:
			if ext in IMAGE_FORMS:
				image2audio(args)
			else:
				codec = ["-sample_fmt", "s16"] if args.format in ("wav", "flac") else ["-c:a", "libopus", "-vbr", "on"] if args.format in ("ogg", "opus") else ["-vbr", "on"]
				args = ffmpeg_start + ("-vn", "-i", args.input, "-f", args.format, *codec, "-b:a", bps_format(args.format), args.output)
				print(args)
				subprocess.run(args)
	finally:
		if input2:
			import os
			os.remove(input2)

if __name__ == "__main__":
	main()