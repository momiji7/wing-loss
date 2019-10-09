import json
import hashlib

def load_txt_file(file_path):
    with open(file_path, 'r') as tfile:
        contents = tfile.readlines()
    tfile.close()
    contents = [x.strip() for x in contents]
    return contents



def generate_json(file_path, save_path):
    contents = load_txt_file(file_path)
    
    data = {}
    print(len(contents))
    for idx, con in enumerate(contents):

        con_split = con.split(' ')
        img_path = con_split[0]
        pts_path = con_split[1]
        box = [float(con_split[2]), float(con_split[3]), float(con_split[4]), float(con_split[5])]
        
        pts_con = load_txt_file(pts_path)
        pts = []
        for i in range(68):
            temp_str = pts_con[3+i].split(' ')
            x = float(temp_str[0])
            y = float(temp_str[1])
            pts.append(x)
            pts.append(y)
        
        landmarks = {}
        landmarks['image_path'] = img_path
        landmarks['num_landmarks'] = 68
        landmarks['landmarks'] = pts
        landmarks['bbox'] = box
        

        fd=open(img_path, "rb")
        fcont=fd.read()
        fmd5=hashlib.md5(fcont)
        #data[fmd5.hexdigest()] = landmarks
        data['{:04d}'.format(idx)] = landmarks
        

    with open(save_path, 'w') as fp:
        json.dump(data, fp)

generate_json('/search/speech/xz/face_alignment/Facetron/data/lists/300W/300w.train.DET', '300w_train_before.json')        
#generate_json('/search/speech/xz/face_alignment/Facetron/data/lists/300W/300w.train.DET', '300w_train.json')
#generate_json('/search/speech/xz/face_alignment/Facetron/data/lists/300W/300w.test.common.DET', '300w_test_common.json')
#generate_json('/search/speech/xz/face_alignment/Facetron/data/lists/300W/300w.test.challenge.DET', '300w_test_challenge.json')
#generate_json('/search/speech/xz/face_alignment/Facetron/data/lists/300W/300w.test.full.DET', '300w_test_full.json')
