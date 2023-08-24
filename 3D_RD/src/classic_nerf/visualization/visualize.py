import cv2
import os
def generate_video(out_folder,vid_name):

    def condition(element):
        a,b=element.split("_")
        a,b=b.split(".")
        b=int(a.strip(" "))
        return b

    image_folder = out_folder 
    video_name = out_folder+vid_name +".mp4v"

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")],key=condition)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()