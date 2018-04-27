import pylab
import numpy as np

import csv

#make it better later.... or not
artists_number = 18745 #from file 
users_number = 2100 # from file
max_users_conections = 25434 # from file

#list_of_files = [('DeathsNVictories_1.txt', 'Time Evolution - simpleRL fullBase'),('DeathsNVictories_1WF.txt', 'Time Evolution - wandercogFull fullBase')]
#list_of_files = [('../lastFM/user_artists.dat', 'simpleRL')]
list_of_files = [('../lastFM/test.dat')]#, ('../lastFM/user_friends_test.dat')]

#datalist = [(pylab.loadtxt(filename), label) for filename, label in list_of_files ]



for file in list_of_files:
	data = np.genfromtxt(file,
                     #skip_header=1,
                     #skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter='	')
	#print(data)

	if file == '../lastFM/test.dat':

		last_user = 2

		temp_all_users_rating = []

		temp_user_rating = np.full((artists_number),0.0) #vetor vazio para rank de artistas por usuario
		#print(temp_user_rating)

		for entry in data:
			user = entry[0]

			if user == last_user:
				temp_user_rating[entry[1]-1] = entry[2] # na pos artist_id ponha o valor count
			else:
				last_user = user # atualiza o user

				temp_all_users_rating.append(temp_user_rating)
				temp_user_rating = np.full((artists_number),0.0) # reinicializa o temp user		
		
				temp_user_rating[entry[1]] = entry[2]#float(entry[2])


		temp_all_users_rating.append(temp_user_rating)


		users_rating = np.asarray(temp_all_users_rating)


		print(users_rating)


		######## norm
		for i in range(0,len(users_rating)):
			#print(users_rating.size)
			max = np.max(users_rating[i,:])
			#print(users_rating[i,:])
			#print(users_rating[i,52]/max)

			users_rating[i,:] = (users_rating[i,:]/max)*5.0
	
			#print(users_rating[i,52])
	
			#print(max)

		np.savetxt("foo.dat", users_rating,fmt='%.4f')


	elif file == '../lastFM/user_friends_test.dat':

		print('sou lindo!')

		#temp_all_users_rating = []

		temp_user_rating = np.full((users_number,users_number),0) #vetor vazio para rank de artistas por usuario
		#print(temp_user_rating)

		for entry in data:
			#user = entry[0]

			#if user == last_user:
			temp_user_rating[entry[0],entry[1]] = 1 #f
		#	else:
		#		last_user = user # atualiza o user

		#		temp_all_users_rating.append(temp_user_rating)
		#		temp_user_rating = np.full((artists_number),0.0) # reinicializa o temp user		
		
		#		temp_user_rating[entry[1]] = entry[2]#float(entry[2])


		#temp_all_users_rating.append(temp_user_rating)


		#users_rating = np.asarray(temp_all_users_rating)
		users_rating = temp_user_rating

		print(users_rating)

		temp_user_weight = np.full((users_number),0.0)
		for i in range(users_number):
			temp_user_weight[i] = sum(users_rating[i,:])/max_users_conections
	##end if
		#print(len(temp_user_weight))
		print(temp_user_weight)
		np.savetxt("foo_friends_matrix.dat", users_rating,fmt='%.1f')
		np.savetxt("foo_friends.dat", temp_user_weight,fmt='%.8f')

##end for





