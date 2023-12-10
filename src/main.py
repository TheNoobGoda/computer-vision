from img_proc import ImgProc
from hand_tracking import HandTrack
from sheet_music import SheetMusic

def main(video_path,save_imgs,see_video,detect_blakc_keys):

    #detect the piano keyboard
    img = ImgProc.get_first_frame(video_path,save_imgs)
    cropped_image,shape = ImgProc.find_piano(img,save_imgs)

    #detect the piano keys
    black_keys, white_keys = ImgProc.find_keys(cropped_image,save_imgs)
    white_keys = SheetMusic.sortKeys(white_keys)
    black_imgs, white_imgs =  ImgProc.get_key(cropped_image,black_keys,white_keys)
    black_keys, white_keys = ImgProc.fix_key_coords(black_keys,white_keys,shape)

    #detect pressed keys in video
    keys= HandTrack.handTrakc(video_path,black_keys,white_keys,black_imgs, white_imgs,see_video,detect_blakc_keys)

    #convert pressed keys to music sheet
    print(SheetMusic.getKeyNotes(black_keys,white_keys,keys))


if __name__ == "__main__":
    main('video/piano_video8.mp4',False,True,False)