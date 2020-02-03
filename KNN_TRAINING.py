import csv
import math
from scipy import stats
import numpy

def hitungpanjangakurasi(sumakurasi):
    return sum(sumakurasi)/len(sumakurasi) #menghitung hasil akurasi berdasarkan split data

def hitungakurasi(z, validasiDataTrain):
    return (z/len(validasiDataTrain)) * 100 #menghitung akurasi yakni dengan jumlah banyaknya data dengan data train yang sudah di validasi dengan kengalikan 100 persen

def pengurangan(x,y): #pengurangan yang ditujukan untuk mencari selisih jarak
    tot = (x-y)**2
    return tot

def jumlahtotdistance(dataTest, dataTrain): #menggunakan prosedur pengurangan yang nantinya akan menghitung berdasarkan nilai x1 hingga x5 pada data test dan data train yang nantinya akan dimasukkan ke dalam rumus euclidan
    hasil = []
    xsatu = pengurangan(dataTest[1], dataTrain[1])
    xdua = pengurangan(dataTest[2], dataTrain[2])
    xtiga = pengurangan(dataTest[3], dataTrain[3])
    xempat = pengurangan(dataTest[4], dataTrain[4])
    xlima = pengurangan(dataTest[5], dataTrain[5])
    hasil = math.sqrt(xsatu + xdua + xtiga + xempat + xlima)
    return hasil

def dataTrain(path):
    data = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader, None)
        for row in spamreader:
            data.append(
                [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])])
        # print(data)
    return data

def cariKelas(KNN, datax): #menentukan kelas mana yang lebih dominan dari hasil perhitungan euclidan
   a = []
   for i in range(KNN):
       a.append(datax[i][1])
   return stats.mode(a)[0] #memakai fungsi mod untuk mencari nilai terbanyak dari kelas yang telah dilakukan splitting

def p(prediksi, test, z): #menghitung nilai prediksi yang ada di kolom 6 pada dataTRrain yakni pendefinisian kelasnya untuk mengecek apakah nilai tersebut sama jika ya akan diberikan spesifikasi kelas yang sesuai dengan data training yang sudah dihitung
    if (prediksi == test[6]):
        z += 1
    return z

def splittingDataTrain():
    dataT = dataTrain('DataTrain_Tugas3_AI.csv') #mendefinisikan dataTRain ke dalam variable dataT
    jk = [] #penyimpanan list hasil akhir
    num_split = 100 #data yang akan dibagi yang ditujukan untuk menghitung data per data total hingga mendapatkan akurasi terbaik
    panjang = len(dataT)-num_split #pengurangan hasil pembanding yang akan dibandingkan dengan num_split yang telah ditentukan
    for fork in range(1,200): #melakukan perulangan yang khendaknya sesuai dengan jumlah data test untuk menentukan k terbaik dari k = 1 hingga 199
        sumakurasi = []
        for i in range(0, panjang, num_split): #perbandingan dilakukan dengan menggunakan perulangan yang nantinya perulangan sendiri itu akan melakukan kelipatan berdasarkan jumlah data awal yang telah di train lalu akan terus menambahkan kelipatannya berdasarkan data split yang awal sudah ditentukan
            z  = 0
            dTrain = list(dataT[1:]) #train data berdasarkan list yang telah tersisia
            validasiDataTrain = dTrain[i: i +10] #validasi dari dTrain
            for j in range(i+10, i-1, -1): #dikarenakan data yang telah di train di awal tidak boleh di lakukan train kembali maka data yang telah di train akan dihapus di dalam array
                dTrain.pop(j)
            for test in validasiDataTrain: #perhitungan jarak yang menggunakan iterasi test yang menentukan akurasinya
                dataHasil= []
                for train in dTrain:
                    y = jumlahtotdistance(test,train)
                    dataHasil.append([y, train[6]])
                dataHasil.sort(key=lambda x: x[0]) #melakukan sorting berdasarkan indeks pertama pada file data hasil yang sudah ditentukan dengan jumlah jarak pada variable y
                prediksi = cariKelas(fork, dataHasil)
                z = p(prediksi, test, z)
            iniakurasi = hitungakurasi(z, validasiDataTrain)
            sumakurasi.append(iniakurasi)
            x= (hitungpanjangakurasi(sumakurasi))
        jk.append(x)
        print('k =', fork, 'akurasi = ', x)
    return jk

if __name__ == "__main__":
    jk = []
    jk = splittingDataTrain()
    numpy.savetxt('akurasi.csv', jk, delimiter=',', fmt='%s') #nilai akurasi akan dimasukkan ke dalam cs untuk mengetahui nilai akurasi terbaik dari k =1 hingga 199
    