class SheetMusic:
    def getKeyNotes(black_keys,white_keys,keys):
        #define note representations for black and white keys
        black_notes = ["C#","D#","F#","G#","A#","C#","D#","F#","G#","A#"]
        white_notes = ["C","D","E","F","G","A","B","C","D","E","F","G","A","B","D"]

        sheet = []
        index = -1
        
        for key in keys:
            index +=1
            #check to see if key is valid
            if index > 0 and index+1 < len(keys) and len(key) == 2:
                if len(keys[index-1]) == 1 and len(keys[index+1]) == 1:
                    if (key[0] == keys[index-1][0] and key[1] == keys[index+1][0]) or (key[1] == keys[index-1][0] and key[0] == keys[index+1][0]):
                        continue

            chord = []

            #map key to its corresponding musical note
            for i in key:
                if i[0] == "b":
                    for j in range(len(black_keys)):
                        #print(i,black_keys[j])
                        if i[1] == black_keys[j]:
                            chord.append(black_notes[j])
                            break
                else:
                    for j in range(len(white_keys)):
                        if i[1] == white_keys[j]:
                            chord.append(white_notes[j])
                            break

            sheet.append(chord)

        return sheet
    
    def sortKeys(keys):
        sorted_keys = []

        #sort the keys based on their starting frame indices
        while len(keys) > 0:
            min = float('inf')
            index = 0

            for i in range(len(keys)):
                if keys[i][0] < min: 
                    min = keys[i][0]
                    index = i

            sorted_keys.append(keys[index])
            keys.remove(keys[index])

        return sorted_keys