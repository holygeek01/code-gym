import pyaudio
def record(BLOCKSIZE):
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE =16000
	CHUNK = 512
	RECORD_SECONDS = BLOCKSIZE
	WAVE_OUTPUT_FILENAME = "file.wav"
	audio = pyaudio.PyAudio()
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,frames_per_buffer=CHUNK)
	print ("recording...")
	frames = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)			
	print "finished recording" 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
#record(5)
print("hey its done")