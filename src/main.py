from img_proc import ImgProc
from hand_tracking import HandTrack

def main():
    ImgProc.get_first_frame('video/piano_video.mp4')
    ImgProc.find_piano('img/results/piano.jpg')
    black_kesy, white_keys = ImgProc.find_keys('img/results/cropped_keyboard.jpg')
    #print(black_kesy, white_keys)
    #HandTrack.handTrakc('video/piano_video.mp4')



if __name__ == "__main__":
    main()