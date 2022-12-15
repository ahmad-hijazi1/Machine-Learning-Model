import pandas as pd #importing data importation package
data = []
data1 = []
data2 = []
training_set = [[45.2, 50, 1, 3.8, 0], [44.4, 50, 2, 3.6, 0], [42.8, 50, 4, 3.2, 0], [43.6, 50, 3, 3.4, 0], [46, 50, 0, 4, 0], [44.5, 50.0, 1.5, 3, 1], [34, 50, 13, 0, 3], 
                [34, 50, 10, 0, 6], [34, 50, 12, 0, 4], [42, 50, 5, 3, 0], [35, 50, 12, 0, 3], [40, 50, 0, 0, 10], [34, 50, 14, 0, 2], [34, 50, 16, 0, 0], [44, 50, 1, 1, 4], 
                [34, 50, 0, 0, 16], [36.4, 50, 12, 1.6, 0], [41.2, 50, 6, 2.8, 0], [38, 50, 10, 2, 0], [34.8, 50, 14, 1.2, 0], [39.6, 50, 8, 2.4, 0], [50, 50, 0, 0, 0], 
                [44.3, 50, 2.1, 3.5, 0.1], [44.5, 50, 1.9, 3.5, 0.1], [44.4, 50, 1.9, 3.6, 0.1], [44.5, 50, 1.8, 3.6, 0.1], [44.5, 50, 1.9, 3.6, 0], [44.6, 50, 1.8, 3.5, 0.1], 
                [44.6, 50, 1.7, 3.6, 0.1], [44.6, 50, 1.8, 3.6, 0], [44.5, 50, 1.8, 3.5, 0.2], [44.5, 50, 1.7, 3.6, 0.2], [44.5, 50, 1.7, 3.7, 0.1], [44.6, 50, 1.7, 3.7, 0]]
for i in range(0, 201, 1):
    x =i/10 #since I did not know how to use a decimal step
    for j in range(0, 51, 1):
        y = j/10
        for k in range(0, 201, 1):
            z = k/10
            b = [round(50-x-y-z,1), 50.0, x, y, z]
            if(50-x-y-z) >= 30 and b not in training_set:
                p = round(50-x-y-z,1)
                data = [p, 50.0, x, y, z, (p*1.09+50*0.79+1.2*x+ 0.99*y+1.12*z)/100, (p*149 +50*176+x*145+y*156+z*169)/100, (p*124 +50*147+x*128+y*126+z*137)/100, 
                        (p*1.91 +50*1.54+x*1.9+y*1.83+z*2.2)/100, (p*10 +50*4+x*11+y*8+z*10)/100, (p*0.325 +50*0.489+x*0.312+y*0.365+z*0.365)/100]
                data1.append(data)
#Saving both csv files
df = pd.DataFrame(data1, columns=('Ni', 'Ti', 'Cu', 'Fe', 'Pd', 'cs', 'arc', 'mr', 'en', 'ven', 'dor'))
df.to_csv('C:/Users/ahmad/Thesis Results November 2022/Search Space Iteration 13.csv')
trset = [[45.2, 50, 1, 3.8, 0, 3.15], [44.4, 50, 2, 3.6, 0, 3.4], [42.8, 50, 4, 3.2, 0, 3.71], [43.6, 50, 3, 3.4, 0, 3.78], [46, 50, 0, 4, 0, 4.21], [44.5, 50.0, 1.5, 3, 1, 4.26],
         [34, 50, 13, 0, 3, 4.70], [34, 50, 10, 0, 6, 5.32], [34, 50, 12, 0, 4, 5.8], [42, 50, 5, 3, 0, 5.83], [35, 50, 12, 0, 3, 5.93], 
         [40, 50, 0, 0, 10, 6.04], [34, 50, 14, 0, 2, 6.65], [34, 50, 16, 0, 0, 7.4], [44, 50, 1, 1, 4, 8.36], [34, 50, 0, 0, 16, 8.53], [36.4, 50, 12, 1.6, 0, 8.62], 
         [41.2, 50, 6, 2.8, 0, 10.16], [38, 50, 10, 2, 0, 10.34], [34.8, 50, 14, 1.2, 0, 10.79], [39.6, 50, 8, 2.4, 0, 12.66], [50, 50, 0, 0, 0, 29.89], [44.3, 50, 2.1, 3.5, 0.1, 4.861143035], 
         [44.5, 50, 1.9, 3.5, 0.1, 4.833926134], [44.4, 50, 1.9, 3.6, 0.1, 4.833660299], [44.5, 50, 1.8, 3.6, 0.1, 4.820051705], [44.5, 50, 1.9, 3.6, 0, 4.819969898], 
         [44.6, 50, 1.8, 3.5, 0.1, 4.820317622], [44.6, 50, 1.7, 3.6, 0.1, 4.806442185], [44.6, 50, 1.8, 3.6, 0, 4.806360898], [44.5, 50, 1.8, 3.5, 0.2, 4.83400828], 
         [44.5, 50, 1.7, 3.6, 0.2, 4.820133418], [44.5, 50, 1.7, 3.7, 0.1, 4.806177168], [44.6, 50, 1.7, 3.7, 0, 4.792486227]]
for l in range(len(trset)):
    p1 = trset[l][0]
    p2 = trset[l][1]
    p3 = trset[l][2]
    p4 = trset[l][3]
    p5 = trset[l][4]
    training_set_with_properties = [p1, p2, p3, p4, p5, (p1*1.09+50*0.79+1.2*p3+ 0.99*p4+1.12*p5)/100, (p1*149 +50*176+p3*145+p4*156+p5*169)/100, (p1*124 +50*147+p3*128+p4*126+p5*137)/100,
                                    (p1*1.91 +50*1.54+p3*1.9+p4*1.83+p5*2.2)/100, (p1*10 +50*4+p3*11+p4*8+p5*10)/100, (p1*0.325 +50*0.489+p3*0.312+p4*0.365+p5*0.365)/100, trset[l][5]]
    data2.append(training_set_with_properties)
df = pd.DataFrame((data2), columns=('Ni', 'Ti', 'Cu', 'Fe', 'Pd', 'cs', 'arc', 'mr', 'en', 'ven', 'dor', 'Delta T'))
df.to_csv('C:/Users/ahmad/Thesis Results November 2022/Training Set Iteration 13.csv') 
