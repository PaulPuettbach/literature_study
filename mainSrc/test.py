import numpy as np
'''
get 8 by 8 array
check if new dim is needed
add if needed
iterate over to add array (has only two iteratable dimensions and add where needed)
eg if diff model 30 we make ( counted from the right the higher number is the one that is rightmost so the i is the rightmost value)
new[row, :, :, collumn] = new[row, :, :, collumn] + to_add[row,collumn]

[0,1,2,3,4,5,6]
third = 3
'''
# def elementwise(array):
#     for i in range(len(array)):
#         if array.ndim == 2:
#             for j in range(len(array[i])):
#                 print(array[i][j])
#         else:
#             elementwise(array[i])


two_zero = np.array([[1,2,5],[3,4,1],[2,4,1,]])
two_one = np.array([[1,2,2],[1,4,1],[2,2,2]])
one_zero = np.array([[4,2,1],[1,4,3],[1,2,3]])
three_zero = np.array([[3,1,4],[6,2,1],[3,4,5]])
new = np.array([one_zero,one_zero,one_zero])

model_i = 2
model_j = 0

#new[row, :, collumn]

string = "new["
for dim in range(new.ndim):
    if dim == new.ndim - model_i -1: # we need the ith position counted from the right
        string += "row"
    elif dim == new.ndim - model_j -1:
        string += "collumn"
    else:
        string += ":"
    if dim == new.ndim -1:
        string += "]"
        break
    string += ","

to_exec = string + "=" + string + "+ two_zero[row,collumn]"
print(to_exec)
for row in range(len(two_zero)):
    for collumn in range(len(two_zero[row])):
        exec(to_exec)

indices = np.where(new == np.amax(new))
ziplist = []
for index in indices:
    ziplist.append(index)
# zip the 2 arrays to get the exact coordinates
listOfCordinates = list(zip(*ziplist))
# travese over the list of cordinates
print(listOfCordinates) # always atleast one

print("this is 1 0 ")
print(one_zero)
print("this is 2 0 ")
print(two_zero)
print("this is the after combining")
print(new)

model_i = 2
model_j = 1


string = "new["
for dim in range(new.ndim):
    if dim == new.ndim - model_i -1: # we need the ith position counted from the right
        string += "row"
    elif dim == new.ndim - model_j -1:
        string += "collumn"
    else:
        string += ":"
    if dim == new.ndim -1:
        string += "]"
        break
    string += ","

to_exec = string + "=" + string + "+ two_one[row,collumn]"
print(to_exec)
for row in range(len(two_one)):
    for collumn in range(len(two_one[row])):
        exec(to_exec)
indices = np.where(new == np.amax(new))
ziplist = []
for index in indices:
    ziplist.append(index)
# zip the 2 arrays to get the exact coordinates
listOfCordinates = list(zip(*ziplist))
# travese over the list of cordinates
print(listOfCordinates) # always atleast one

print("this is 2 1 ")
print(two_one)
print("this is the after combining")
print(new)

model_i = 3
model_j = 0

new = np.array([new,new,new])

print("this is the new")
print(new)


string = "new["
for dim in range(new.ndim):
    if dim == new.ndim - model_i -1: # we need the ith position counted from the right
        string += "row"
    elif dim == new.ndim - model_j -1:
        string += "collumn"
    else:
        string += ":"
    if dim == new.ndim -1:
        string += "]"
        break
    string += ","

to_exec = string + "=" + string + "+ three_zero[row,collumn]"
print(to_exec)
for row in range(len(three_zero)):
    for collumn in range(len(three_zero[row])):
        exec(to_exec)

print("this is 3 0 ")
print(three_zero)
print("this is the after combining")
print(new)

num_topics = 3
chosen_permutations = []
for i in range(num_topics):
    indices = np.where(new == np.amax(new))
    ziplist = []
    for index in indices:
        ziplist.append(index)
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(*ziplist))
    # travese over the list of cordinates
    print("first set of numbers")
    print(listOfCordinates) # always atleast one
    chosen_permutations.append(listOfCordinates[0])

    for indx in range(new.ndim):
        string = "new["
        for dim in range(new.ndim):
            if dim == indx:
                string += str(listOfCordinates[0][dim])
            else:
                string += ":"
            if dim == new.ndim -1:
                string += "]"
                break
            string += ","
        string = string + " = 0"
        exec(string)

print("final all")
print(chosen_permutations) # always atleast one
    

print("this is the new \n", new)
