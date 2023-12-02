from img_proc import ImgProc
from hand_tracking import HandTrack
from sheet_music import SheetMusic

def main():
    ImgProc.get_first_frame('video/piano_video.mp4')
    _,shape = ImgProc.find_piano('img/results/piano.jpg')
    black_keys, white_keys = ImgProc.find_keys('img/results/cropped_keyboard.jpg')
    black_keys, white_keys = ImgProc.fix_key_coords(black_keys,white_keys,shape)
    #print(black_kesy, white_keys)
    keys= HandTrack.handTrakc('video/piano_video.mp4',black_keys,white_keys)
    print(SheetMusic.getKeyNotes(black_keys,white_keys,keys))


if __name__ == "__main__":
    main()