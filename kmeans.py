from typing import List
import random
class KMeansClusterClassifier:
    def __init__(self, cluster):
        self.cluster = cluster
        self.distortion=0
    def euclidean(self,centroids, X:List[List[float]], dim):
        y = [0]*len(X)
        summ = 0
        # Her bir centroid için sırayla her bir satır üzerinden öklid uzaklığı hesaplanır
        euclidean = [0]*len(centroids) # hesaplanan öklid uzaklıklarını tutan array
        for i in range(len(X)):
            for j in range(len(centroids)):
                for k in range(dim): # attribute sayısı kadar döner
                    summ += (X[i][k] - centroids[j][k])**2
                summ = summ ** (1/2)
                euclidean[j] = summ
                summ=0
            # her centroid için hesaplanan değerlerin minimumuna bakılır. Minimum değer hangi centroide ait ise onun indexi y değerine atılır.
            minV = 10000
            for j in range(len(euclidean)):
                if(euclidean[j] < minV):
                    minV = euclidean[j]
                    index = j
            y[i] = index
            euclidean = [0]*len(centroids)
        return y
    # rastgele merkez atamamdaki kordinat sınırlarını belirlememe yardımcı olan metot.
    # her sütundaki maksimum değeri arrayde tutar.
    def maxFaetureValues(self,X: List[List[float]]):
        maxValues = [0]*len(X[0])
        maxV=-1
        for i in range(len(X[0])):
            for j in range(len(X)):
                if X[j][i] > maxV:
                    maxV = X[j][i]
            maxValues[i]=maxV 
            maxV=-1
        return maxValues

    # rastgele nokta atamamda da elimizdeki class sayısı kadar ayrım istiyorum. Bu metot buna yardımcı olacaktır.
    # metot y arrayindeki unique değerleri verir ör: 0,2,3,3,3,5,0,2,2,2 ->> 0,2,3,5
    def unique(self, y):
        uniq = []
        for i in y:
            boo = True
            for j in uniq:
                if (i == j):
                    boo = False
            if (boo):
                uniq.append(i)
        return uniq

    def fit(self, X: List[List[float]], y: List[int]):
        iterations=100
        centroids=[0]*self.cluster
        centroidsXYZ = []
        maxValues = self.maxFaetureValues(X)
        unq=0
        # centroidlerimin hepsi en az bir classı ayırsın istiyorum. o yüzden ayırana kadar while döngüsü döner.
        # ör: 0,1,2 classlarımız var ben sadece 0,1 leri ayırmış merkezler istemiyorum.
        while(unq==0):
            for i in range(self.cluster):
                for j in range(len(X[0])):
                    num = float(maxValues[j]) # her column için maks değer üzerinden random sayımı döndürüyorum (merkezler için)
                    x = random.uniform(0,num)
                    centroidsXYZ.append(x) # kordinat değerleri tuttar [x,y,z,d] gibi
                centroids[i] = centroidsXYZ # gelen centroid merkezlerini tutar [[x,y,z,d],...,[xn,yn,zn,dn]] gibi
                centroidsXYZ=[]
            self.centroid = centroids.copy()
            points = self.euclidean(centroids,X,len(X[0])) #y değerleri
            control = self.unique(points)
            # her class için ayrım olmuş mu diye bakar
            if(len(control)==self.cluster):
                break
        coordinate=[]
        # merkezler güncellenir
        for i in range(iterations): #iterasyon sayımız kadar döner
            centroids = []
            for j in range(self.cluster):
                   coordinate = self.mean(j,X,points) # o cluster için noktalar arası ortalama hesabı yapar.
                   centroids.append(coordinate)
            points = self.euclidean(centroids, X, len(X[0])) # bir daha sonuç değerleri güncellerin
        self.centroid = centroids # en son centroidler güncellenir
        self.distortion = self.sse(X,points) # elbow point için
        return points

    #elbow için sum of squeare errorlara bakılır.
    def sse(self,X:List[List[float]],y:List[float]):
        sums = 0
        for i in range(len(self.centroid)):
            for j in range(len(y)):
                if y[j]==i:
                    for k in range(len(X[0])):
                        sums += (X[j][k] -self.centroid[i][k]) ** 2
        return sums**(1/2)

    # gelen cluster için ve clusterın sınıfındaki her noktanın ortalamasını alaraktan merkez kordinatını günceller.
    def mean(self,clusterNum, X:List[List[float]], y:List[float]):
        coordinate=[]
        sum=0
        count=0
        control=0
        for j in range(len(X[0])):
            for i in range(len(X)):
                if clusterNum==y[i]:
                    sum+=X[i][j]
                    count+=1
            if(count != 0):
                sum=float(sum/count)
                coordinate.append(sum)
            if(count==0 and control==0):
                control=1
                for k in range(len(X[0])):
                    coordinate.append(self.centroid[clusterNum][k])
            sum=0
            count=0
        return coordinate

    def predict(self, X: List[List[float]]):
        return self.euclidean(self.centroid, X, len(X[0]))


