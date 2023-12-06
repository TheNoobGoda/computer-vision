from img_proc import ImgProc
from hand_tracking import HandTrack
from sheet_music import SheetMusic

def main():
    ImgProc.get_first_frame('video/piano_video3.mp4')
    _,shape = ImgProc.find_piano('img/results/piano.jpg')
    black_keys, white_keys = ImgProc.find_keys('img/results/cropped_keyboard.jpg')
    black_imgs, white_imgs =  ImgProc.get_key('img/results/cropped_keyboard.jpg',black_keys,white_keys)
    black_keys, white_keys = ImgProc.fix_key_coords(black_keys,white_keys,shape)
    keys= HandTrack.handTrakc('video/piano_video3.mp4',black_keys,white_keys,black_imgs, white_imgs)
    print(SheetMusic.getKeyNotes(black_keys,white_keys,keys))


if __name__ == "__main__":
    main()