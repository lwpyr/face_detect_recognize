from DAR import *
from config import config, update_config
import cv2, os, shutil, time

class FaceDAR(object):
    def __init__(self, conf_path):
        
        update_config(conf_path)

        self.detector = eval(config.DETECT_MODEL).FaceD(config=config.DETECT)
        self.aligner = eval(config.ALIGN_MODEL).FaceA(config=config.ALIGN)

        if not config.DATABASE_READY:
            self.process_id_folder(config.ID_FOLDER, config.ALIGNED_FOLDER)

        self.recognizer = eval(config.RECOGNIZE_MODEL).FaceR(config=config.RECOGNIZE)
        self.recognizer.load_id_files(config.ALIGNED_FOLDER)

    def test(self, img_path, frame_id=0):
        im = cv2.imread(img_path)
        bboxes = self.detector.Detect(im)
        patches = self.aligner.Align(im, bboxes)
        names = self.recognizer.Recognize(patches)
        i = 0
        for name, patch in zip(names, patches):
            #if name != 'null':
            save_path = './result/%s' % name
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            patch = patch.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(save_path, str(frame_id) + str(i) + '.jpg'), patch)
            i += 1


    def process_folder(self, folder_path, vis_score=False):
        img_folder_path = os.path.join(folder_path, 'imgs')
        if not os.path.exists(img_folder_path):
            os.mkdir(img_folder_path)
        files = os.listdir(folder_path)
        for f in files:
            if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG') or f.endswith('.png') or f.endswith('.PNG'):
                shutil.move(os.path.join(folder_path, f), os.path.join(img_folder_path, f))
        
        result_folder = os.path.join(folder_path, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
            for person in self.recognizer.db_person:
                person_path = os.path.join(result_folder, person)
                if not os.path.exists(person_path):
                    os.mkdir(person_path)
            detect_result_path = os.path.join(result_folder, 'detection')
            if not os.path.exists(detect_result_path):
                    os.mkdir(detect_result_path)
        
        img_files = os.listdir(img_folder_path)
        img_files.sort()
        max_count = 0
        res_dict = dict()
        for img_file in img_files:
            t1 = time.time()
            img_path = os.path.join(img_folder_path, img_file)
            im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            t2 = time.time()
            pre_time = t2-t1

            t1 = time.time()
            bboxes = self.detector.Detect(im)
            t2 = time.time()
            bboxes, patches = self.aligner.Align(im, bboxes)
            t3 = time.time()
            names, scores = self.recognizer.Recognize(patches)
            t4 = time.time()
            d_time = t2 - t1
            a_time = t3 - t2
            r_time = t4 - t3

            if len(bboxes)>max_count:
                max_count = len(bboxes)

            t1 = time.time()
            if vis_score:
                detect_res = self.detector.vis_dets(im, bboxes, names, scores)
            else:
                detect_res = self.detector.vis_dets(im, bboxes, names)
            cv2.imwrite(os.path.join(result_folder, 'detection', 'dets_'+img_file), detect_res)
            for idx, (name, patch) in enumerate(zip(names, patches)):
                if name != 'null':
                    save_path = os.path.join(result_folder, name, img_file)
                    cv2.imwrite(save_path, patch)
                    res_dict[name] = res_dict.get(name, 0) + 1
            t2 = time.time()
            post_time = t2-t1
            print('%s: PRE-%.3f sec,  DAR-%.3f sec [%.3f, %.3f, %.3f], POST-%.3f sec.' % (img_file, pre_time, (d_time+a_time+r_time), d_time, a_time, r_time, post_time))

        info = '%s: detect-%d, recognize-%d' % (folder_path, max_count, len(res_dict))
        print(info)
        with open(os.path.join(result_folder, 'log.txt'), 'w') as fh:
            fh.write(info)
    
    def process_id_folder(self, folder_path='./id_img', aligned_path='./id_img_aligned'):

        if not os.path.exists(folder_path):
            print('Database miss!')
            exit(1)
        if not os.path.exists(aligned_path):
            os.mkdir(aligned_path)

        files = os.listdir(folder_path)
        for f in files:
            print("processing: %s" % f)
            img_path = os.path.join(folder_path, f)
            aligned_img_path = os.path.join(aligned_path, f)

            im = cv2.imread(img_path, cv2.IMREAD_COLOR)
            bboxes = self.detector.Detect_raw(im)
            bboxes, patches = self.aligner.Align(im, bboxes)
            assert len(patches) > 0, "Database failed! [%s]" % f
            cv2.imwrite(aligned_img_path, patches[0])
        self.detector.reset()
