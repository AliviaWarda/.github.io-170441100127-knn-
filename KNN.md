# k-Nearest Neighbor (k-NN) Algoritma



![](D:\Template\mkdocs-material-master\docs\assets\images\gambar judul.png)

## A. Pengertian k-Nearest Neighbor (k-NN)

*K-Nearest Neighbor* (k-NN atau KNN) adalah suatu metode untuk melakukan [klasifikasi](https://id.wikipedia.org/wiki/Pengenalan_pola) terhadap objek berdasarkan data pembelajaran yang jaraknya paling dekat dengan objek tersebut. Tujuan nya adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan training sample. 

Data baru yang diklasifikasi selanjutnya diproyeksikan pada ruang dimensi banyak yang telah memuat titik-titik c data pembelajaran. Proses klasifasikasi dilakukan dengan mencari titik **c** terdekat dari **c-baru** (*nearest neighbor*)*.*Teknik pencarian tetangga terdekat yang umum dilakukan dengan menggunakan formula jarak euclidean*.*


## B. Kelebihan dan Kekurangan k-Nearest Neighbor (k-NN)

​	Kelebihan : 

1. Sangat sederhana implementasi

2. Kuat dalam hal ruang pencarian

3. Efektif untuk menghitung data dalam skala kecil

4. Beberapa parameter untuk acuan : jarak metric,k

Kekurangan : 

1. Perlu untuk menentukan nilai k yang optimal sehingga untuk menyatakan jumlah tetangga terdekatnya lebih mudah

2. Biaya komputasi yang cukup tinggi karena perhitungan jarak harus dilakukan pada setiap query instance

## C. Algoritma k-Nearest Neighbor (k-NN)

Langkah-Langkah algoritma k-Nearest Neighbor(k-NN)

1. Menentukan parameter k (jumlah tetangga paling dekat).

2. Menghitung kuadrat jarak eucliden objek terhadap data training yang diberikan.

3. Mengurutkan hasil no 2 secara *ascending* (berurutan dari nilai tinggi ke rendah)

4. Mengumpulkan kategori Y (Klasifikasi nearest neighbor berdasarkan nilai k)

5. Dengan menggunakan kategori nearest neighbor yang paling mayoritas maka dapat dipredisikan kategori objek.

Contoh Perhitungan k-Nearest Neighbor(k-NN)

​	Diberikan data Training berupa dua atribut Bad dan Good untuk mengklasiikasikan sebuah data apakah tergolong Bad atau Good,berikut ini adalah contoh datanya :

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar soal.png)

Langkah penyelesaian : 

**Pertama**, Kita tentukan parameter K. Misalnya kita buat jumlah tertangga terdekat **K = 3**.

**Ke-dua**, kita hitung jarak antara data baru dengan semua data training. Kita menggunakan *Euclidean Distance*. Kita hitung seperti pada table berikut :

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar no.2.PNG)

**Ke-tiga**, kita urutkan jarak dari data baru dengan data training dan menentukan tetangga terdekat
berdasarkan jarak minimum K.

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar no.3.PNG)

Dari kolom 4
(urutan jarak) kita mengurutkan dari yang terdekat ke terjauh antara jarak data baru dengan data training. ada 2 jarak yang sama (yaitu 4) pada data baris 2 dan baris 6, sehingga memiliki urutan yang sama. Pada kolom 5 (Apakah termasuk 3-NN?) maksudnya adalah K-NN menjadi 3-NN , karena nilai K ditentukan sama
dengan 3.
**Ke-empat**, tentukan kategori dari tetangga terdekat. Kita perhatikan baris 3, 4, dan 5 pada gambar sebelumnya (diatas). Kategori Ya diambil jika nilai **K<=3**. Jadi baris 3, 4, dan 5 termasuk kategori Ya dan sisanya Tidak.

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar no.4.PNG)

Kategori ya untuk K-NN pada kolom 6, mencakup baris 3,4, dan 5. Kita berikan kategori berdasarkan tabel awal. baris 3 memiliki kategori Bad, dan 4,5 memiliki kategori Good.

**Ke-lima**, gunakan kategori mayoritas yang sederhana dari tetangga yang terdekat tersebut sebagai nilai prediksi data yang baru.

 ![](D:\Template\mkdocs-material-master\docs\assets\images\gambar n0.5.PNG)

Data yang kita miliki pada baris 3, 4 dan 5 kita punya 2 kategori Good dan 1 kategori Bad. Dari jumlah mayoritas (**Good > Bad**) tersebut kita simpulkan bahwa data baru (**X=3 dan Y=5**) termasuk dalam kategori **Good**.

## E. Software Requirement

Python 3.0 atau versi yang lebih baru, disini saya menggunakan python 3.7

1. IDE Pycharm

2. Library Python yang digunakan:

- Numpy

Numpy merupakan sebuah library pada Python yang berfungsi untuk melakukan operasi vektor dan matriks dengan mengolah array dan array multidimensi. Biasanya NumPy digunakan untuk kebutuhan dalam menganalisis data.

instal numpy:

```
pip install numpy 
```

- Pandas

pandas adalah sebuah librari berlisensi BSD dan open source yang menyediakan struktur data dan analisis data yang mudah digunakan dan berkinerja tinggi untuk bahasa pemrograman Python.

instal pandas:

```
pip install pandas
```

- Matplotlib

Matplotlib adalah *library* paling banyak digunakan oleh *data science* untuk menyajikan datanya ke dalam visual yang lebih baik.

instal matplotlib:

```
pip install matplotlib
```

- Scikit Learn

Machine learning ada yang berbasis statistika ada juga yang tidak. Salah satunya adalah support vector machine dan regresi linier. Mungkin bagi sebagian orang sudah biasa menulis sendiri library untuk implementasi kedua algoritma tadi. Tapi untuk membuatnya dalam waktu singkat tentu butuh waktu yang tidak sedikit pula.

Scikit-Learn memberikan sejumlah fitur untuk keperluan data science seperti:

- Algoritma Regresi
- Algoritma Naive Bayes
- Algoritma Clustering
- Algoritma Decision Tree
- Parameter Tuning
- Data Preprocessing Tool
- Export / Import Model
- Machine learning pipeline dan lainnya

instal Scikit Learn :

```
pip install scikit-learn 
```

## F. Implementasi k-Nearest Neighbor (k-NN)

Import library yang dibutuhkan 

Mengimportkan library untuk mendukung implementasi k-Nearest Neighbor 

```
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline
```

Memuat Dataset 

Mengimport dataset yang digunakan untuk pengimplemtasian k means clustering

dataset bisa didownload [disini](https://github.com/yusrilx02/K-Mean-Clustering-E-Commerce-Data-Actual-transactions-from-UK-retailer/blob/master/data.csv)

```
from sklearn.datasets import load_iris
iris = load_iris()
```

```
type(iris)
```

> ```OUTPUT 
> sklearn.utils.Bunch
> ```

Data Preprocessing 

Melihat Bagaimana data dipecah 

```
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))
```

```output 
X_train shape: (112, 4)
y_train shape: (112,)
X_test shape: (38, 4)
y_test shape: (38,)
```

Melihat hubungan antar variabel kode warna berdasarkan jenis spesies

```
sns.pairplot(df.drop(labels=['Id'], axis=1), hue='Species')
```

```
<seaborn.axisgrid.PairGrid at 0x7f35f35fc4e0>
```

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar variabel 1.png)

Plot data dan klasifikasi untuk melihat data memiliki klasifikasi

```
plt.figure(figsize=[18,8])
plt.scatter(data1['Species'], data1['sepal length (cm)'],  marker= 'o')
plt.scatter(data1['Species'], data1['sepal width (cm)'], marker= 'x')
plt.scatter(data1['Species'], data1['petal width (cm)'], marker= '*')
plt.scatter(data1['Species'], data1['petal length (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.xlabel('Species Name')
plt.show()
```

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar plot 1.png)

```
plt.figure(figsize=[18,8])
plt.plot(data1['sepal length (cm)'], marker= 'o')
plt.plot(data1['sepal width (cm)'], marker= 'x')
plt.plot(data1['petal length (cm)'], marker= '*')
plt.plot(data1['petal width (cm)'], marker= ',')
plt.ylabel('Length in cm')
plt.legend()
plt.show()
```

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar grafik plot 2.png)

Dari plot di atas, muncul tren pengelompokan elemen data.
Tujuan dari latihan pembelajaran mesin ini:
Bunga berdasarkan pengukuran fisiknya, diklasifikasikan menjadi spesies tertentu. Artinya, ada hubungan kapal antara pengukuran fisik dan spesies. Kita perlu membuat model / metode yang digunakan untuk pengukuran tertentu kita harus dapat mengklasifikasikan spesies. Dari dataset yang diberikan, pembelajaran mesin terjadi untuk menentukan hubungan dan sebuah model dibuat untuk memprediksi spesies tersebut .

Menambahkan  kolom 'Spesies' ke Dataset dengan klasifikasi ini

```
def categorize(a):
    if a == 0.0:
        return('setosa')
    if a == 1.0:
        return('versicolor')
    return('virginica')
data1['Species'] = data1['target'].apply(categorize)
```

```
data1.head()
```

![](D:\Template\mkdocs-material-master\docs\assets\images\gambar nambah tabel.PNG)









Link References : 

https://medium.com/@alfiindah/k-nearest-neighbors-dengan-python-dan-scikit-learn-f5fda40b4e76

<https://www.kaggle.com/susree64/k-nearest-neighbor-with-iris-data-set>

<https://informatikalogi.com/algoritma-k-nn-k-nearest-neighbor/>

<https://id.wikipedia.org/wiki/KNN>

<https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e>

<https://www.advernesia.com/blog/data-science/pengertian-dan-cara-kerja-algoritma-k-nearest-neighbours-knn/>

