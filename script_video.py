from moviepy.editor import *

def make_clip(imgs, name):
	clip_2d = ImageSequenceClip(imgs, fps=30).margin(10)
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

NUMBER_RUN = 128
NAME = "dim64_2"
clip_img = make_clip(["{}_{}.png".format(NAME, i) for i in range(NUMBER_RUN)],"")
clip_fft = make_clip(["{}_fft_{}.png".format(NAME, i) for i in range(NUMBER_RUN)],"")
clip = clips_array([[clip_img, clip_fft]])
save_video(clip, "dim2")