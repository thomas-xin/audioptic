import contextlib
import os
import subprocess
import cv2 as cv
import imageio_ffmpeg as ii
import numpy as np
ffmpeg = ii.get_ffmpeg_exe()
ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-v", "info", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "none")
C0 = 440 / 32 * 2 ** (3 / 12)


class AutoDeleter(contextlib.AbstractContextManager):

	def __init__(self, files=()):
		self.files = set(files)

	def append(self, v):
		self.files.add(v)

	def __enter__(self):
		return self

	def __exit__(self, *args):
		for f in self.files:
			if os.path.exists(f):
				try:
					os.remove(f)
				except Exception:
					pass
		self.files.clear()
		return


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
	"icns",
	"qoi",
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
	fn = f"{ctx.output.rsplit('.', 1)[0]}~.pcm"
	args = ffmpeg_start + ("-vn", "-i", ctx.input, "-f", "f32le", "-ac", "2", "-ar", str(ctx.sample_rate), fn)
	ctx.deleter.append(fn)
	subprocess.run(args)
	data = np.fromfile(fn, dtype=np.float32)
	data = data.reshape((len(data) // 2), 2).T
	transformed = transform(data)
	del data

	ima = np.empty((transformed.shape[1], transformed.shape[2] * 2, 3), dtype=np.float32)
	left_amp, right_amp = np.abs(transformed)
	left_pha, right_pha = np.angle(transformed)
	del transformed
	ima[:, ::2, 0] = left_pha * (180 / np.pi) + 180
	del left_pha
	ima[:, 1::2, 0] = right_pha * (180 / np.pi) + 180
	del right_pha
	left_exp = np.clip((np.ceil(np.log2(left_amp)) + BASE) * (1 / MAX), 0, 1)
	right_exp = np.clip((np.ceil(np.log2(right_amp)) + BASE) * (1 / MAX), 0, 1)
	left_frac = left_amp / 2 ** (left_exp * MAX - BASE)
	del left_amp
	right_frac = right_amp / 2 ** (right_exp * MAX - BASE)
	del right_amp
	ima[:, ::2, 1] = left_exp
	del left_exp
	ima[:, 1::2, 1] = right_exp
	del right_exp
	ima[:, ::2, 2] = left_frac
	del left_frac
	ima[:, 1::2, 2] = right_frac
	del right_frac
	cvt = cv.cvtColor(ima, cv.COLOR_HLS2BGR)
	del ima
	cvt *= 255
	im = np.round(cvt).astype(np.uint8)
	del cvt
	if ctx.format in ("png", "webp", "tiff", "jpg", "bmp"):
		cv.imwrite(ctx.output, im)
	else:
		fn = f"{ctx.output.rsplit('.', 1)[0]}~.bgr"
		with open(fn, "wb") as f:
			f.write(im.data)
		ctx.deleter.append(fn)
		args = ffmpeg_start + ("-f", "rawvideo", "-video_size", f"{im.shape[1]}x{im.shape[0]}", "-pix_fmt", "bgr24", "-r", "1", "-i", fn, "-vframes", "1", "-b:v", "9999999", "-quality", "100", "-lossless", "1", ctx.output)
		print(args)
		del im
		subprocess.run(args)

def image2audio(ctx):
	if ctx.extension not in ("png", "webp", "tiff", "jpg", "bmp"):
		fn = f"{ctx.output.rsplit('.', 1)[0]}~.bmp"
		args = ffmpeg_start + ("-i", ctx.input, "-vframes", "1", "-pix_fmt", "bgr24", fn)
		print(args)
		subprocess.run(args)
		ctx.input = fn
		ctx.deleter.append(fn)
	im = cv.imread(ctx.input)
	assert im is not None, "Image was not loaded properly!"
	im1 = im.astype(np.float32)
	del im
	im1 *= 1 / 255
	ima = cv.cvtColor(im1, cv.COLOR_BGR2HLS)
	del im1
	left_pha = (ima[:, ::2, 0] - 180) * (np.pi / 180)
	right_pha = (ima[:, 1::2, 0] - 180) * (np.pi / 180)
	left_amp = ima[:, ::2, 2] * 2 ** (np.round(ima[:, ::2, 1] * MAX) - BASE)
	right_amp = ima[:, 1::2, 2] * 2 ** (np.round(ima[:, 1::2, 1] * MAX) - BASE)
	left_amp[ima[:, ::2, 1] == 0] = 0
	right_amp[ima[:, 1::2, 1] == 0] = 0
	del ima
	left = left_amp * np.exp(1j * left_pha, dtype=np.complex64)
	del left_amp, left_pha
	right = right_amp * np.exp(1j * right_pha, dtype=np.complex64)
	del right_amp, right_pha
	transformed = np.stack([left[:, :right.shape[1]], right])
	del left, right

	data = np.clip(itransform(transformed), -2, 2)
	del transformed
	codec = ["-sample_fmt", "s16"] if ctx.format in ("wav", "flac") else ["-c:a", "libopus", "-vbr", "on"] if ctx.format in ("ogg", "opus") else ["-vbr", "on"]
	fn = f"{ctx.output.rsplit('.', 1)[0]}~.pcm"
	with open(fn, "wb") as f:
		f.write(data.T.tobytes())
	del data
	ctx.deleter.append(fn)
	args = ffmpeg_start + ("-f", "f32le", "-ac", "2", "-ar", str(ctx.sample_rate), "-i", fn, "-f", ctx.format, *codec, "-b:a", bps_format(ctx.format), ctx.output)
	print(args)
	subprocess.run(args)


try:
	from importlib.metadata import version
	__version__ = version("audioptic")
except Exception:
	__version__ = "0.0.0-unknown"

def convert(ctx):
	ctx.deleter = deleter = AutoDeleter()
	with deleter:
		if is_url(ctx.input):
			fn = ctx.input.split("?", 1)[0].rsplit("/", 1)[-1]
			if "." in fn:
				name, ext = fn.rsplit(".", 1)
				fn = name + "~." + ext
			else:
				fn += "~"
			import asyncio
			import streamshatter
			print(fn)
			f = asyncio.run(streamshatter.shatter_request(ctx.input, filename=fn))
			if f:
				f.close()
			ctx.input = fn
			deleter.append(fn)
		name = ctx.input.rsplit(".", 1)[0]
		try:
			import filetype
			ext = filetype.guess(ctx.input).extension
		except AttributeError:
			ext = ctx.input.rsplit(".", 1)[-1]
		ctx.extension = ext
		fmt = ctx.format
		if not fmt:
			fmt = ctx.format = ctx.output.rsplit(".", 1)[-1] if "." in ctx.output else ("opus" if ext in IMAGE_FORMS else "webp")
		if not ctx.output:
			ctx.output = f"{name}.{fmt}"
		if fmt in IMAGE_FORMS:
			if ext in IMAGE_FORMS:
				args = ffmpeg_start + ("-i", ctx.input, "-vframes", "1", "-b:v", "9999999", "-quality", "100", "-lossless", "1", ctx.output)
				print(args)
				subprocess.run(args)
			else:
				audio2image(ctx)
		else:
			if ext in IMAGE_FORMS:
				image2audio(ctx)
			else:
				codec = ["-sample_fmt", "s16"] if ctx.format in ("wav", "flac") else ["-c:a", "libopus", "-vbr", "on"] if ctx.format in ("ogg", "opus") else ["-vbr", "on"]
				args = ffmpeg_start + ("-vn", "-i", ctx.input, "-f", ctx.format, *codec, "-b:a", bps_format(ctx.format), ctx.output)
				print(args)
				subprocess.run(args)

def main():
	import argparse
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
	convert(args)

if __name__ == "__main__":
	main()