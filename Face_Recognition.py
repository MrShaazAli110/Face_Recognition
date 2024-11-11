import face_recognition
import cv2
import numpy as np
import os 
from datatime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("photos/job.jpg") #loading the photo of people to recognition
job_encoding = face_recognition.face_encoding(jobs_image)[0]

Abdul_Kalam = face_recognition.load_image_file("photos/Abdul Kalam.jpg")
job_encoding = face_recognition.face_encoding(Abdul_Kalam_image)[0]

Amitabh_Bachchan = face_recognition.load_image_file("photos/Amitabh Bachchan.jpg")
job_encoding = face_recognition.face_encoding(Amitabh_Bachchan_image)[0]

Salman_Khan = face_recognition.load_image_file("photos/Salman Khan.jpg")
job_encoding = face_recognition.face_encoding(Salman_Khan_image)[0]

Shahrukh_khan = face_recognition.load_image_file("photos/shahrukh khan.jpg")
job_encoding = face_recognition.face_encoding(Shahrukh_khan_image)[0]

Virat_Kohli = face_recognition.load_image_file("photos/Virat Kohli.jpg")
job_encoding = face_recognition.face_encoding(Virat_Kohli_image)[0]







known_face_encoding = [
jobs_encoding,
Abdul_Kalam_encoding,
Amitabh_Bachchan_encoding, 
Salman_Khan_encoding,
Shahrukh_khan_encoding,
Virat_Kohli_encoding
]

known_face_names = [
    "jobs"
    "Abdul Kalam",
    "Amitabh Bachchan",
    "Salman Khan",
    "Shahrukh Khan",
    "Virat Kohli"
]

students = known_face_names.copy()


face_locations = []
face_encodings = []
face_names = []
s=True


now = datetime.now()
current_date: object = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',nextline = '')
inwriter = csv.writer(f)


while True:
    _,frame = video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([name,current_time])
    cv2.imshow("Attendenceee System",frame)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
