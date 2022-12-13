import numpy as np
'''
get 8 by 8 array
check if new dim is needed
add if needed
iterate over to add array (has only two iteratable dimensions and add where needed)
eg if diff model 30 we make

new[i, :, :, j] = new[i, :, :, j] + to_add[i,j]

'''
def elementwise(array):
    for i in range(len(array)):
        if array.ndim == 2:
            for j in range(len(array[i])):
                print(array[i][j])
        else:
            elementwise(array[i])


AC = np.array([[1,2],[3,4]])
BC = np.array([[1,2],[1,4]])
AB = np.array([[4,2],[1,4]])
new = np.array([AB,AB])
test = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

print("this is the to test \n", test)
elementwise(test)

# print("this is the to add \n", AC)
# print("this is the old \n", AB)
# print("this is the new \n", new)

# print("this is test", new[0,:,:])




