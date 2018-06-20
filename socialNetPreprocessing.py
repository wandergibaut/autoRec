import numpy as np


# make it better later.... or not
artists_number = 18745  # from file
users_number = 2100  # from file
max_users_conections = 25434  # from file


list_of_files = [('../lastFM/fooData/foo.dat'),
                 ('../lastFM/fooData/friends_matrix_weights.dat')]


data = np.genfromtxt('../lastFM/user_artists.dat',
                     # skip_header=1,
                     # skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter='	')

user_friends_data = np.genfromtxt('../lastFM/fooData/friends_matrix_weights.dat',
                                  dtype=None,
                                  delimiter=' ')

social_data_input = np.full((users_number, artists_number * 2), 0.0)

last_user = 2

# vetor vazio para rank de artistas por usuario
temp_user_rating = np.full((artists_number), 0.0)
# print(temp_user_rating)
user_rating_matrix = np.full((users_number, artists_number), 0.0)

for entry in data:
    user = entry[0] - 1

    if user == last_user:
        # na pos artist_id ponha o valor count
        temp_user_rating[entry[1] - 1] = entry[2]
    else:
        last_user = user  # atualiza o user

        user_rating_matrix[user, :] = np.copy(temp_user_rating)

        # reinicializa o temp user
        temp_user_rating = np.full((artists_number), 0.0)
        temp_user_rating[entry[1]] = entry[2]  # float(entry[2])


user_rating_matrix[user, :] = np.copy(temp_user_rating)


user_r_matrix = np.asarray(user_rating_matrix)

# norm
for i in range(0, len(user_r_matrix)):
                        # print(users_rating.size)
    max = np.max(user_r_matrix[i, :])
    if max != 0:
        user_r_matrix[i, :] = (user_r_matrix[i, :] / max) * 5.0


for user in range(len(user_r_matrix)):
    if np.max(user_r_matrix[user, :]) > 0:
        # ve o amigo com o maior peso e pega o indicie dele
        index = np.argmax(user_friends_data[user, :], 0)
        half_1 = np.copy(user_r_matrix[user, :])
        half_2 = np.copy(user_r_matrix[index, :])
        full = np.append(half_1, half_2)

        social_data_input[user, :] = np.copy(full)
        # one half + other half


np.savetxt("../lastFM/fooData/social_autoRec_input.dat",
           social_data_input, fmt='%.4f')

np.savetxt("../lastFM/fooData/foo_with_zeros.dat", user_r_matrix, fmt='%.4f')
