import numpy as np
from IPython import embed
def store_json_rank( list_images, name_file ):
    """
    create json rank for image id
    """
    json_file =  os.path.join( "{}.json".format(name_file) )
    with open( json_file, 'wb' ) as f:
        N = list_images.shape[0]
        for i, name_ima in enumerate(list_images):
            line=''
            if i == 0:
                line+="[{\n\"image\" :\"{}\",\n".format(name_ima)
            else:
                line+="{\n\"id\" :\"{}\",\n".format(i)
            line+="\"Image\":\"%s\"\n" % name
            if i == N-1:
                line+="}]"
            else:
                line+="},\n"

            # write
            f.write(line)
    return json_file

if __name__ == '__main__':
    data = np.loadtxt( 'paris/qimlist.txt', dtype='str' )[:,0]

    with open( 'qimlist.json', 'w' ) as f:
        for i, name in enumerate(data):
            line = ''
            if i==0:
                line+='[{\n'
            else:
                line+='{\n'
            line+="\"image\":\"{}\",\n".format(name)
            line+="\"id\":\"{}\"\n".format(i)

            if i == data.shape[0]-1:
                line+='}]\n'
            else:
                line+='},\n'
            f.write(line)
