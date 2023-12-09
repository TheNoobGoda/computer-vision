from img_proc import ImgProc
from hand_tracking import HandTrack
from sheet_music import SheetMusic

def main(video_path,save_imgs):
    img = ImgProc.get_first_frame(video_path,save_imgs)
    cropped_image,shape = ImgProc.find_piano(img,save_imgs)
    black_keys, white_keys = ImgProc.find_keys(cropped_image,save_imgs)
    white_keys = SheetMusic.sortKeys(white_keys)
    black_imgs, white_imgs =  ImgProc.get_key(cropped_image,black_keys,white_keys)
    black_keys, white_keys = ImgProc.fix_key_coords(black_keys,white_keys,shape)
    keys= HandTrack.handTrakc(video_path,black_keys,white_keys,black_imgs, white_imgs)
    print(SheetMusic.getKeyNotes(black_keys,white_keys,keys))


if __name__ == "__main__":
    main('video/piano_video8.mp4',False)