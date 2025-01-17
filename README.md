# MaskDetection
## Overview
Proyek ini bertujuan untuk mengembangkan sistem deteksi objek yang mampu mengidentifikasi orang yang memakai masker dan yang tidak memakai masker secara real-time. Sistem ini dirancang untuk mendukung upaya pengawasan dan penegakan protokol kesehatan, terutama dalam situasi pandemi atau di area dengan risiko penyebaran penyakit yang tinggi. 

## Dataset
Dataset didapatkan dari gabungan dataset [Roboflow Mask Wearing iOS Computer Vision Project](https://universe.roboflow.com/mohamed-traore-2ekkp/roboflow-mask-wearing-ios/dataset/16) yang memiliki gambar pemakaian masker yang bervariasi. Persebaran data yang digunakan adalah sebagai berikut:  
| Folder  | Image Count | 
| ------------- | ------------- |
| Train  | 2475 | 
| Validation | 684 | 
| Test | 369 |

## Model 
Proyek ini menggunakan model YOLOv11 sebagai algoritma deteksi objek utama, yang telah diimplementasikan dengan melakukan hyperparameter tuning untuk mengoptimalkan performa deteksi.

### Environment
- Quadro RTX 5000
- Python 3.12.1
- Pytorch 2.5.1
- Torchvision 0.20.1
- Torchaudio 2.5.1
- Ultralytics 8.3.39

### Hyper-parameter Tuning
Hyper-parameter tuning dilakukan dengan menggunakan fungsi model.tune yang disediakan pada library ultralytics untuk mencari learning rate yang terbaik. Tuning dilakukan sebanyak 25 iterasi dengan masing-masing iterasi dijalankan selama 100 epoch.

### Metrik Evaluasi
![results](https://github.com/user-attachments/assets/327d332a-e1cd-4dbd-89cb-78fa7e6cf0da)

| Model | epoch  | Imgsz | lr0  | lrf | Recall  | Precision | mAP50  | mAP50-95 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Baseline | 100  | 1280  | 0.01  | 0.01 | 0.88338 | 0.93452  | 0.93328 | 0.6848  |

### Hasil
![image](https://github.com/user-attachments/assets/9fed85e6-26fb-470d-82b9-38b8d6720da0)







