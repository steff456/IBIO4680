from create_texton_representation import create_texton_representation
from classify_KNN import classify_KNN

#Main menu of the application
a = True
train_textons = []
test_textons = []
k = 50
routes = ['./100x100 - 10t/texton_representation.npy', './100x100 - 20t/texton_representation.npy',
'./100x100 - 30t/texton_representation.npy', './200x200 - 10t/texton_representation.npy',
'./200x200 - 20t/texton_representation.npy', './200x200 - 30t/texton_representation.npy']

while(a):
    print('Please choose your option: ')
    print('1. Create Texton Representation for Data Directory - Only run in high memory device')
    print('2. Classify Nearest Neighbour')
    print('3. Classify Random Forest')
    print('4. Exit')
    a = raw_input()
    if a == '1':
        #Without changing any parameter
        train_textons, test_textons = create_texton_representation()
    elif a == '2':
        print('1. 100x100 Windows: 10 train images')
        print('2. 100x100 Windows: 20 train images')
        print('3. 100x100 Windows: 30 train images')
        print('4. 200x200 Windows: 10 train images')
        print('5. 200x200 Windows: 20 train images')
        print('6. 200x200 Windows: 30 train images')
        print('7. new~!')
        print('8. Calculate all!')
        print('9. Cancel')
        b = raw_input('Select Database')
        if b == '1':
            route = routes[0]
            p = classify_KNN(k, route)
        elif b == '2':
            route = routes[1]
            p = classify_KNN(k, route)
        elif b == '3':
            route = routes[2]
            p = classify_KNN(k, route)
        elif b == '4':
            route = routes[3]
            p = classify_KNN(k, route)
        elif b == '5':
            route = routes[4]
            p = classify_KNN(k, route)
        elif b == '6':
            route = routes[5]
            p = classify_KNN(k, route)
        elif b == '7':
            p = classify_KNN(k, '', train_textons, test_textons)
        elif b == '8':
            p = []
            for route in routes:
                p_act = classify_KNN(k, route)
                p.append(p_act)
            print(p)
        elif b == '9':
            continue
    elif a == '3':
        print('1. 100x100 Windows: 10 train images')
        print('2. 100x100 Windows: 20 train images')
        print('3. 100x100 Windows: 30 train images')
        print('4. 200x200 Windows: 10 train images')
        print('5. 200x200 Windows: 20 train images')
        print('6. 200x200 Windows: 30 train images')
        print('7. new~!')
        print('8. Calculate all!')
        print('9. Cancel')
        b = raw_input('Select Database')
        if b == '1':
            route = routes[0]
            p = classify_RF(k, route)
        elif b == '2':
            route = routes[1]
            p = classify_RF(k, route)
        elif b == '3':
            route = routes[2]
            p = classify_RF(k, route)
        elif b == '4':
            route = routes[3]
            p = classify_RF(k, route)
        elif b == '5':
            route = routes[4]
            p = classify_RF(k, route)
        elif b == '6':
            route = routes[5]
            p = classify_RF(k, route)
        elif b == '7':
            p = classify_RF(k, '', train_textons, test_textons)
        elif b == '8':
            p = []
            for route in routes:
                p_act = classify_RF(k, route)
                p.append(p_act)
            print(p)
        elif b == '9':
            continue
    elif a == '4':
        print('Bye!')
        break
    
    a = True

