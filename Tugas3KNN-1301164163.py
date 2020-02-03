import csv
import math
from scipy import stats
import numpy
import  KNN_TRAINING

def pengurangan(x,y): #pengurangan yang digunakan untuk menghitung selisih jarak yang nantinya akan dimasukkan ke dalam rumus euclidian
    tot = (x-y)**2
    return tot

def jumlahtotdistance(dataTest, dataTrain): #perhitungan euclidian
    hasil = []
    xsatu = pengurangan(dataTest[1], dataTrain[1])
    xdua = pengurangan(dataTest[2], dataTrain[2])
    xtiga = pengurangan(dataTest[3], dataTrain[3])
    xempat = pengurangan(dataTest[4], dataTrain[4])
    xlima = pengurangan(dataTest[5], dataTrain[5])
    hasil = math.sqrt(xsatu + xdua + xtiga + xempat + xlima)
    return hasil


def dataTrain(path): #untuk memasukkan nilai data train ke dalam variable
    data = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader, None)
        for row in spamreader:
            data.append(
                [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])])
        # print(data)
    return data

def dataTest(path): #untuk memasukkan nilai data test ke dalam variable
    data = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader, None)
        for row in spamreader:
            data.append(
                [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), row[6]])
        # print(data)
    return (data)

# def vote(distance, dataTrain, KNN):
#     terpilih = 0
#     tidak = 0
#     for i in range(0, KNN):
#         if dataTrain[distance[i][1]][6] == "1":
#             terpilih += 1
#         else:
#             tidak += 1
#     if  (terpilih>tidak):
#         return 1
#     else:
#         return 0

def cariKelas(KNN, datax): #menentukan kelas untuk data yang telah ditentukan perhitunga euclidian
   a = []
   for i in range(KNN):
       a.append(datax[i][1])
   return stats.mode(a)[0] #menggunakan libary statistik untuk menentukan nilai terbesar yang akan dijadikan kelas yang ditentukan untuk data test

if __name__ == "__main__":
    x =[]
    k = 13 #nilai k yang sudah ditentukan dalam knn_training.py
    dTrain = dataTrain('DataTrain_Tugas3_AI.csv')
    dTest = dataTest('DataTest_Tugas3_AI.csv')
    for i in range(200): #200 berdasarkan data yang berada pada dTest
        dataHasil= []
        for j in range(800): #800 berdasarkan data training
            y = jumlahtotdistance(dTest[i],dTrain[j]) #menghitung dengan menggunakan rumus euclidan
            dataHasil.append([y, dTrain[j][6]]) #memasukkan nilai pada indeks ke 6 yakni kelas ke dalam data hasil
        dataHasil.sort(key=lambda x: x[0])
        print(dataHasil[:5]) #mengeluarkan nilai dataHasil dari awal hingga indeks ke 5
        x.append(cariKelas(k, dataHasil))
        print(cariKelas(k, dataHasil))
    numpy.savetxt('TebakanTugas3.csv', x, delimiter=',', fmt='%s')
