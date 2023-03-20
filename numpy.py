import numpy

print("Using numpy:\n")
# 1.creating 2 matrices
p = numpy.array([[1,0],[0,1]])
q = numpy.array([[1,2],[3,4]])
# 2.perform multiplication
mul = numpy.matmul(p,q)
# 3.perform addition
add = numpy.add(p,q)
# 4.print the result.
print("Multiplication:\n",mul)
print("Addition:\n",add)
# 5.checking matrix multiplication same as a dot product for the above.
print("Same:\n",mul == p.dot(q))


print("\n\nWithout using numpy:\n")
# 1.creating 2 matrices
p = [[1,0],[0,1]]
q = [[1,2],[3,4]]
# 2.perform multiplication
mul = [[0 for x in range(len(p))] for y in range(len(q[0]))] # initially we are taking the mul  matrix with zeros as values.
for i in range(len(p)):
    for j in range(len(q[0])):
        for k in range(len(q)):
            mul[i][j] += p[i][k] * q[k][j]  # updating the matrix values
# 3.perform addition
add = [[0 for x in range(len(p))] for y in range(len(q[0]))]
for i in range(len(p)):
        for j in range(len(p[0])):
                add[i][j] =  p[i][j] + q[i][j]
# 4.printing the result
print("Multiplication:\n",mul)
print("Addition:\n",add)