import pylab
import numpy as np


artists_number = 18745 #from file 


#list_of_files = [('DeathsNVictories_1.txt', 'Time Evolution - simpleRL fullBase'),('DeathsNVictories_1WF.txt', 'Time Evolution - wandercogFull fullBase')]
#list_of_files = [('../lastFM/user_artists.dat', 'simpleRL')]
list_of_files = [('../lastFM/test.dat')]

#datalist = [(pylab.loadtxt(filename), label) for filename, label in list_of_files ]


data = np.genfromtxt('../lastFM/test.dat',
                     #skip_header=1,
                     #skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter='	')
print(data)

#fase_1_process

last_user = 2

temp_all_users_rating = []

temp_user_rating = np.full((artists_number),0) #vetor vazio para rank de artistas por usuario
#print(temp_user_rating)

for entry in data:
	#user = entry(0)
	#print('entry: ' + str(entry))
	user = entry[0]
	#print('user: ' + str(user))

	if user == last_user:
		temp_user_rating[entry[1]] = entry [2] # na pos artist_id ponha o valor count
	else:
		last_user = user # atualiza o user
		print('new user: ' + str(user))


		temp_all_users_rating.append(temp_user_rating)
		#print(temp_all_users_rating)
		#users_rating[user, :] = users_rating


		temp_user_rating = np.full((artists_number),0) # reinicializa o temp user		
		temp_user_rating[entry[1]] = entry[2]
		
		#print(entry[1])
		print(np.max(temp_user_rating))
		#print(entry[2])
		#print(temp_user_rating[entry[1]])

temp_all_users_rating.append(temp_user_rating)


users_rating = np.asarray(temp_all_users_rating)

print(np.count_nonzero(temp_user_rating))
print(np.count_nonzero(users_rating))
	#print(temp_user_rating)
	#print(user)

#print(temp_user_rating)
print(users_rating)


######## norm
#print(np.max(users_rating))
#print(users_rating[0,52])

#caraios = users_rating[0,:] 

#print(np.max(caraios))
#print(np.sum(users_rating[0,:]))

#print(np.max(temp_all_users_rating))
#print(temp_all_users_rating[0][51])

for i in range(0,len(users_rating)):
	#print(users_rating.size)
	max = np.max(users_rating[i,:])
	#print(users_rating[i,:])
	users_rating[i,:] = (users_rating[i,:]/max)*5
	print(users_rating[i,51])
	print(max)



