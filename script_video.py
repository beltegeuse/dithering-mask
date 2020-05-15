from moviepy.editor import *

def make_clip(imgs, name):
	clip_2d = ImageSequenceClip(imgs, fps=5).margin(10)
	if name != "":
		txt = TextClip(name, font='Amiri-regular', color='white',fontsize=24)
		txt = txt.set_pos(lambda t: ((clip_2d.w / 2) - txt.w / 2, clip_2d.h - 50))
		final_clip_2d = CompositeVideoClip([clip_2d, txt]).set_duration(clip_2d.duration)
		return final_clip_2d
	else:
		return clip_2d

def save_video(clip, name):
	# Visually lossless (x4 compression rate compared to lossless)
	# Veryslow: Quite depending in term of CPU usage
	clip.write_videofile("{}.mp4".format(name), codec='libx264', ffmpeg_params=[ '-preset', 'veryslow', '-crf', '0'], bitrate=None)

NUMBER_RUN = 32
NAME = ""
clip_img_2d = make_clip(["dim128_2_{}.png".format(i) for i in range(NUMBER_RUN)],"")
clip_fft_2d = make_clip(["dim128_2_fft_{}.png".format(i) for i in range(NUMBER_RUN)],"")
clip_img_1d = make_clip(["dim128_1_{}_.png".format(i) for i in range(NUMBER_RUN)],"")
clip_fft_1d = make_clip(["dim128_1_fft_{}.png".format(i) for i in range(NUMBER_RUN)],"")
clip = clips_array([[clip_img_1d, clip_fft_1d],
	[clip_img_2d, clip_fft_2d]])
save_video(clip, "dim2")
