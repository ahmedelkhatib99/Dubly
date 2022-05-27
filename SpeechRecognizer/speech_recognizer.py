
# import library
import speech_recognition as sr
import moviepy.editor as mp
import os
from pathlib import Path

class SpeechRecognizer:
    def get_text_of_audio(self, video_name):
        # Extract audio from video
        self.input_folder = os.path.join(
            os.path.dirname(__file__), "..\\demo\\input")
        my_clip = mp.VideoFileClip(os.path.join(self.input_folder,video_name))

        audio_name = video_name.split('.')[0]
        my_clip.audio.write_audiofile(os.path.join(self.input_folder,audio_name+'.wav'))
        my_clip.audio.write_audiofile(os.path.join(self.input_folder,audio_name+'.mp3'))

        r = sr.Recognizer()
        with sr.AudioFile(os.path.join(self.input_folder,audio_name+'.wav')) as source:
            audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text, language='es-AR')
            print('Converting audio transcripts into text ...')
            print(text)
        except Exception as e:
            print('Sorry.. run again...', e)
        return [text] , audio_name +'.mp3'


if __name__ == "__main__":
    speech_recognizer_model = SpeechRecognizer("videotest_2.mp4")
    speech_recognizer_model.get_text_of_audio(SpeechRecognizer.AUDIO_FILE_NAME)
