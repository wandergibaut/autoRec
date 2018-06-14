import pylab
import numpy as np

import csv

#make it better later.... or not
artists_number = 18745 #from file 
users_number = 2100 # from file
max_users_conections = 25434 # from file


list_of_files = [('../lastFM/user_friends.dat')]#('../lastFM/user_artists.dat'), ('../lastFM/user_friends.dat')]




for file in list_of_files:
	data = np.genfromtxt(file,

                     names=True,
                     dtype=None,
                     delimiter='	')
	#print(data)

	if file == '../lastFM/user_artists.dat':

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


		#print(users_rating)


		######## norm
		for i in range(0,len(users_rating)):

			max = np.max(users_rating[i,:])


			users_rating[i,:] = (users_rating[i,:]/max)*5.0
	


		np.savetxt("../lastFM/fooData/foo.dat", users_rating,fmt='%.4f')


	elif file == '../lastFM/user_friends.dat':


		#temp_all_users_rating = []

		temp_user_friend_matrix = np.full((users_number,users_number),0.0) #vetor vazio para rank de artistas por usuario
		#print(temp_user_rating)

		for entry in data:

			temp_user_friend_matrix[entry[0]-1,entry[1]-1] = 1 #f


		users_friend_matrix = temp_user_friend_matrix

		#print(users_friend_matrix)

		temp_user_weight = np.full((users_number),0.0)
		for i in range(users_number):
			temp_user_weight[i] = sum(users_friend_matrix[i,:])/max_users_conections
	##end if
		#print(len(temp_user_weight))

		user_friend_weights = np.copy(users_friend_matrix)
		best_friend_index = np.full((users_number),0.0)

		for i in range(users_number):
			for j in range(users_number):
				if user_friend_weights[i,j] == 1:
					user_friend_weights[i,j] = temp_user_weight[j]
					user_friend_weights[j,i] = temp_user_weight[i]

		for i in range (users_number):
			best_friend_index[i] = np.argmax(user_friend_weights[i,:])

		#print(temp_user_weight)
		np.savetxt("../lastFM/fooData/friends_matrix.dat", users_friend_matrix,fmt='%.1f')
		np.savetxt("../lastFM/fooData/friends_weights.dat", temp_user_weight,fmt='%.8f')
		np.savetxt("../lastFM/fooData/friends_matrix_weights.dat", user_friend_weights,fmt='%.8f')
		#uso
		np.savetxt("../lastFM/fooData/best_friend_index.dat", best_friend_index,fmt='%.1f')

##end for





