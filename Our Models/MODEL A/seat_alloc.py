seats=[]
cols=["C1","C2","C3"]
letters=["A","B","C","D","E","F","G","H","I","J"]

seats.append("C1|B33")
seats.append("C3|B13")
for x in letters[2:]:
    seats.append(f'C1|{x}34')

