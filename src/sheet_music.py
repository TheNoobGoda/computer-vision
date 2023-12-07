class SheetMusic:
    def getKeyNotes(black_keys,white_keys,keys):
        black_notes = ["C#","D#","F#","G#","A#","C#","D#","F#","G#","A#"]
        white_notes = ["C","D","E","F","G","A","B","C","D","E","F","G","A","B","D"]
        sheet = []
        for key in keys:
            chord = []
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