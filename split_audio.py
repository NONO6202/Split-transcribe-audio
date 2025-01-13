import librosa
import shutil
import soundfile as sf
import os
import whisper
import csv
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))
model = whisper.load_model("large-v3-turbo").to("cuda")

folders = [item for item in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, item)) and item.isdigit()]
if not folders: largest_folder = 0
else: largest_folder = int(max(folders, key=int))+1
    
rename_count = largest_folder
for element in os.listdir(filepath):
    if os.path.splitext(element)[1] == '.wav':
        os.rename(f"{filepath}/{element}", f"{filepath}/{rename_count:05d}.wav")
        rename_count += 1
        
def average_db(audio_path, sr=22050):
    """오디오 파일의 평균 데시벨을 계산"""
    y, sr = librosa.load(audio_path, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms)
    return np.mean(rms_db)

def split_audio(name_count, min_segment_len=0.66):
    """오디오 분할"""
    input_file = f"{filepath}/{name_count:05d}.wav"
    output_dir = f"{filepath}/{name_count:05d}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y, sr = librosa.load(input_file, sr=22050, mono=True)
    
    avg_db = average_db(input_file, sr=sr)
    print(avg_db)
    silence_threshold = avg_db - 10
    
    intervals = librosa.effects.split(y, top_db=-silence_threshold, frame_length=int(sr * 0.01), hop_length=int(sr * 0.01))

    segment_count = 1
    total_time = 0
    for interval in intervals:
        start_time = interval[0] / sr
        end_time = interval[1] / sr
        segment_length = end_time - start_time

        if segment_length >= min_segment_len:
            segment = y[interval[0]:interval[1]]
            output_file_name = os.path.join(output_dir, f"{segment_count}.wav")
            sf.write(output_file_name, segment, sr)
            print(f"Saved segment: {output_file_name} ({segment_length:.2f} seconds)")
            segment_count += 1
            total_time += segment_length
    
    shutil.copy(os.path.join(filepath, f"{name_count:05d}.wav"), os.path.join(output_dir, f"Origin{name_count:05d}.wav"))
    os.remove(input_file)
    return total_time

total_time = 0
for i in range(largest_folder,largest_folder+rename_count):
    total_time += split_audio(i)
    output_dir = f"{filepath}/{i:05d}"
    
    """분할된 오디오파일을 Whisper모델로 전사"""
    csv_file_path = os.path.join(output_dir, "transcription.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['파일명', '텍스트'])

        for segment_file in os.listdir(output_dir):
            if segment_file.endswith(".wav") and not segment_file.startswith("Origin"):
                audio_filepath = os.path.join(output_dir, segment_file)
                result = model.transcribe(audio_filepath, language="ko")
                transcription = result["text"]
                csv_writer.writerow([segment_file, transcription])
                
    print(f"{i:05d}.wav Completion!!\n")
print(f"Total: {total_time} seconds")