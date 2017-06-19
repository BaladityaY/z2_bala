
"""
python kzpy3/caf/kaffe/y2016/m3/kaffe3_new_start.py
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.                                                        
matplotlib.use('Agg')

from kzpy3.vis import *
from kzpy3.misc.progress import *
from google.protobuf import text_format

from os import listdir
from os.path import isfile, join
import scipy.misc

caffe_root = '/u/vis/x1/bala/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()

model_folders = ['z2','z2_color','z2_color_long','z1_color','z2_color_small_ip1']

def get_imgs(model_folder, sess_ind):
    cam_data_folder = opj(home_path,'caffe/models/'+model_folder+'/camera_data/')
    
    stereo_imgs = scipy.misc.imread(cam_data_folder+'route_{}_pic_0.png'.format(sess_ind))
    
    if model_folder == 'z2_color':
        xshape = stereo_imgs.shape[0]/2
        yshape = stereo_imgs.shape[1]/2
        color_img_data = np.zeros((12,xshape,yshape))
    elif model_folder == 'z2_color_long':
        xshape = stereo_imgs.shape[0]/10
        yshape = stereo_imgs.shape[1]/2
        color_img_data = np.zeros((60,xshape,yshape))
    elif model_folder == 'z2_color_small_ip1':
        xshape = stereo_imgs.shape[0]/2 # timepoints
        yshape = stereo_imgs.shape[1]/2 # eyes
        color_img_data = np.zeros((12,xshape,yshape)) # t*e*rgb


    if model_folder == 'z2':
        n_frames = 1
    elif model_folder == 'z2_color':
        n_frames = 2
    elif model_folder == 'z2_color_long':
        n_frames = 10
    elif model_folder == 'z2_color_small_ip1':
        n_frames = 2
            
    ctr = 0
    for c in range(3):
        for camera in ('left','right'):
            for t in range(n_frames):
                if camera == 'left':
                    color_img_data[ctr,:,:] = stereo_imgs[t*xshape:(t+1)*xshape, 0:yshape, c]
                elif camera == 'right':
                    color_img_data[ctr,:,:] = stereo_imgs[t*xshape:(t+1)*xshape, yshape:, c]
                ctr += 1

    return color_img_data


def print_solver(solver):

    print("")
    
    for l in [(k, v[0].data.shape) for k, v in solver.net.params.items()]:
        print(l)

    print("")
    for l in [(k, v.data.shape) for k, v in solver.net.blobs.items()]:
        if 'split' not in l[0]:
            print(l)


def setup_solver(solver_file_path):
    solver = caffe.SGDSolver(solver_file_path)
    print_solver(solver)
    return solver

def get_net(MODEL_NUM = 0):
    model_folder = model_folders[MODEL_NUM]
    model_path = opj(home_path,'caffe/models/',model_folder)
    print model_path
    net_fn   = opj(model_path,'deploy.prototxt')
    param_fn = opj(model_path,'model.caffemodel')

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))
    net = caffe.Net('tmp.prototxt', param_fn, caffe.TEST)
    #                       #mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
    #                       #channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB 
    print model_folders[MODEL_NUM]
    for n in net.blobs.keys():
        print (np.shape(net.blobs[n].data),n)
    return net

## BALA: replaced karl's get_net
'''
def get_net(model_path,net_fn,param_fn):
    model_path = opj(home_path,'caffe/models',model_folder)
    net_fn   = opj(model_path,'deploy.prototxt')
    param_fn = opj(model_path,'model.caffemodel')
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    solver = setup_solver(net_fn)
    #solver = caffe.Classifier('tmp.prototxt', param_fn,
    #                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
    #                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    print fname(model_path)
    for n in solver.net.blobs.keys():
        print (np.shape(solver.net.blobs[n].data),n)
    return solver
'''

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(solver, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - solver.transformer.mean['ZED_data_pool2']
def deprocess(solver, img):
    return np.dstack((img + solver.transformer.mean['ZED_data_pool2'])[::-1])



def objective_L2(dst):
    dst.diff[:] = dst.data 


def make_step(net, step_size=0.1, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2, input_data_mask=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob ## removed 'solver.'
    dst = net.blobs[end] ## removed 'solver.'

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    net.forward(end=end) ## BALA: changed start from conv1, removed start altogether ## removed 'solver.'  
    objective(dst)  # specify the optimization objective
    
    net.backward(start=end) ## BALA: changed end from data, removed end altogether ## removed 'solver.'  

    if input_data_mask == None:
        g = src.diff[0]
    else:
        g = src.diff[0] * input_data_mask ## BALA: added input_data_mask to the multiplier

    print shape(g)
    print((g.max(),g.min()))
    # apply normalized ascent step to the input image
    g_abs_mean = np.abs(g).mean()
    if not g_abs_mean == 0:
        src.data[:] += (step_size/g_abs_mean) * g 
    else:
        print('np.abs(g).mean() == 0')

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    """    
    if clip:
        bias = solver.net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    
    """



def do_it4(model_folder,dst_path,layer,net,iter_n,start=0,node=-1,input_data=None,drive_mode=None,one_eye=None,one_timept=None,sorted_info=None,ip2_masks=None): ## replaced solver param with net param

    #transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
    #transformer.set_transpose('data', (2,0,1))
    #transformer.set_mean('data', np.load(opj(home_path,'caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1)) # mean pixel
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    layer_shape=list(np.shape(net.blobs[layer].data));## removed 'solver.'  
    layer_shape[0] = 1
    layer_shape = tuple(layer_shape)
    img_path = opj(home_path,dst_path,model_folder+'/'+layer.replace('/','-'))

    unix('mkdir -p ' + img_path)

    if sorted_info == None:
        node_list = range(start,layer_shape[1]) #(num_nodes): ## Karl calls channels nodes 
    else:
        sorted_inds = sorted_info[0]
        unsorted_comp = sorted_info[1]
        node_list = sorted_inds[:25] # [:50]

    if not (ip2_masks == None):
        node_list = range(len(ip2_masks))

    for n_ind in range(len(node_list)):
        n = node_list[n_ind]

        mask7 = np.zeros(layer_shape)
    
        #n = np.random.randint(1000)

        ## BALA: changed this to sample only some individual nodes' gradients
        #mask7[:,n] = 1.0
        '''
        if node >= 0:
            temp_smask = np.zeros(layer_shape[2]*layer_shape[3])
            temp_mask[node] = 1.0
            temp_mask = temp_mask.reshape((layer_shape[2],layer_shape[3]))
            mask7[:,n,:,:] = temp_mask
        else:
            mask7[:,n] = 1.0 
        '''

        if layer == 'conv1':
            samps_per_int = 1 #4 ##BALA: changed this
        
            for xnode in range(layer_shape[2]):
                for ynode in range(layer_shape[3]):
                    if xnode % samps_per_int == 0 and ynode % samps_per_int == 0:
                        mask7[:,n,xnode,ynode] = 1.0

        elif layer == 'conv2':
            samps_per_int = 1 #2 ##BALA: changed this

            for xnode in range(layer_shape[2]):
                for ynode in range(layer_shape[3]):
                    if xnode % samps_per_int == 0 and ynode % samps_per_int == 0:
                        mask7[:,n,xnode,ynode] = 1.0

        elif layer == 'ip2':
            if ip2_masks == None:
                samps_per_int = 1
                mask7[:,n] = 1.0

                if not (sorted_info == None):
                    for node_i in node_list:
                        mask7[:,node_i] = 1.0
            else:
                samps_per_int = 1
                mask7[:,:] = ip2_masks[n]
                print 'at {}, the ip2_mask: {}'.format(n, ip2_masks[n])

        else:
            samps_per_int = 1
            mask7[:,n] = 1.0

            if not (sorted_info == None):
                for node_i in node_list:
                    mask7[:,node_i] = 1.0      

            ''' ## BALA: removed this to mask all nodes in parallel, as above
            if not (sorted_info == None):
                for old_n_ind in range(n_ind):
                    old_n = node_list[old_n_ind]
                    mask7[:,old_n] = 1.0
            '''     
    
        #samps_per_int = node ## BALA: removed this
        
        def objective_kz7(dst):
            if sorted_info == None:
                dst.diff[:] = dst.data * mask7 ##BALA
            else:
                dst.diff[:] = unsorted_comp * dst.data * mask7 ##BALA

        if model_folder == 'z2':
            n_frames = 1
        elif model_folder == 'z2_color':
            n_frames = 2
        elif model_folder == 'z2_color_long':
            n_frames = 10
        elif model_folder == 'z2_color_small_ip1':
            n_frames = 2
        elif model_folder == 'z1_color':
            n_frames = 1

        net.blobs['data'].data[0,:,:,:] = 0.0
        print 'model_folder: {}'.format(model_folder)
        if input_data == None: ## BALA: removed n_ind == 0
            if 'z2' in model_folder:
                print 'in z2 stuff'

                input_data_mask = np.zeros(net.blobs['data'].data[0,:,:,:].shape)
                ctr = 0
                for c in range(3):
                    for camera in ('left','right'):
                        for t in range(n_frames):
                            if one_timept == None or one_timept == t: # only show valid timepoints
                                input_data_mask[ctr,:,:] = 1.0
                                net.blobs['data'].data[0,ctr,:,:] = 255*np.random.random((net.blobs['data'].shape[2], net.blobs['data'].shape[3])) ## BALA: removed 'solver.'

                            if one_eye == 'right' and camera == 'left': # set left eye to zero if only right shown
                                net.blobs['data'].data[0,ctr,:,:] = 0.0
                                input_data_mask[ctr,:,:] = 0.0 
                            elif one_eye == 'left' and camera == 'right': # set right eye to zero if only left shown
                                net.blobs['data'].data[0,ctr,:,:] = 0.0
                                input_data_mask[ctr,:,:] = 0.0 

                            ctr += 1

            elif 'z1' in model_folder:
                print 'in z1 stuff'

                input_data_mask = np.ones(net.blobs['data'].data[0,:,:,:].shape)
                net.blobs['data'].data[0,:,:,:] = 255*np.random.random((3,net.blobs['data'].shape[2], net.blobs['data'].shape[3])) # 3 b/c RGB

        else: ## BALA: removed n_ind == 0
            net.blobs['data'].data[0,:,:,:] = input_data[:,:,:]
            input_data_mask = np.ones(net.blobs['data'].data[0,:,:,:].shape)

        print 'one_eye: {} and one_timept: {}'.format(one_eye, one_timept)
        print 'input data: {}'.format(net.blobs['data'].data[0,:,:,:])

        if not (drive_mode == None):
            print 'changed drive meta: {}'.format(drive_mode)
            net.blobs['metadata'].data[0,:,:,:] = 0.0 #first resets the metadata
            net.blobs['metadata'].data[0,drive_mode,:,:] = 1.0
            net.blobs['metadata'].data[0,1,:,:] = 1.0
        

        pb = ProgressBar(iter_n)

        for i in range(iter_n):
            make_step(net,end=layer,objective=objective_kz7,jitter=1,input_data_mask=input_data_mask)
            src = net.blobs['data'] ## removed 'solver.'  

            if np.mod(i,100.0)==0:
                pb.animate(i+100)

                mi(z2o(net.blobs['data'].data[0,0,:,:]));pause(0.01) ## removed 'solver.'  

        print((model_folder,layer,n))
        vis = net.blobs['data'].data[0,:,:,:] ## removed 'solver.'  
        '''
        for dim in range(12):
            temp_vis = vis[dim,:,:]
            vis[dim,:,:] = 255*(temp_vis - np.amin(temp_vis))/np.amax(temp_vis)
        vis = np.uint8(vis)
        '''

        xshape = vis.shape[1]
        yshape = vis.shape[2]
    
        if model_folder == 'z2':
            n_frames = 1
        elif model_folder == 'z2_color':
            n_frames = 2
        elif model_folder == 'z2_color_long':
            n_frames = 10
        elif model_folder == 'z2_color_small_ip1':
            n_frames = 2
        elif model_folder == 'z1_color':
            n_frames = 1

        if 'z2' in model_folder:
            img = zeros((xshape*n_frames,yshape*2,3))

            ctr = 0
            for c in range(3):
                for camera in ('left','right'):
                    for t in range(n_frames):
                        if camera == 'left':
                            img[t*xshape:(t+1)*xshape, 0:yshape, c] = vis[ctr,:,:]
                        elif camera == 'right':
                            img[t*xshape:(t+1)*xshape, yshape:, c] = vis[ctr,:,:]
                        ctr += 1

        elif 'z1' in model_folder:
            img = zeros((xshape*n_frames,yshape,3))

            for c in range(3):
                img[:,:,c] = vis[c,:,:]

        '''
        #left_t1
        img[0:xshape,0:yshape,0] = vis[0,:,:] #R
        img[0:xshape,0:yshape,1] = vis[4,:,:] #G
        img[0:xshape,0:yshape,2] = vis[8,:,:] #B

        #right_t1
        img[0:xshape,yshape:,0] = vis[1,:,:]
        img[0:xshape,yshape:,1] = vis[5,:,:]
        img[0:xshape,yshape:,2] = vis[9,:,:]

        #left_t2
        img[xshape:,0:yshape,0] = vis[2,:,:]
        img[xshape:,0:yshape,1] = vis[6,:,:] 
        img[xshape:,0:yshape,2] = vis[10,:,:]
    
        #right_t2
        img[xshape:,yshape:,0] = vis[3,:,:]
        img[xshape:,yshape:,1] = vis[7,:,:]
        img[xshape:,yshape:,2] = vis[11,:,:]
        '''
        '''
        img[0:xshape,0:yshape,0] = vis[0,:,:]
        img[0:xshape,0:yshape,1] = vis[1,:,:]
        img[0:xshape,0:yshape,2] = vis[2,:,:]

        img[0:xshape,yshape:,0] = vis[3,:,:]
        img[0:xshape,yshape:,1] = vis[4,:,:]
        img[0:xshape,yshape:,2] = vis[5,:,:]

        img[xshape:,0:yshape,0] = vis[6,:,:]
        img[xshape:,0:yshape,1] = vis[7,:,:]
        img[xshape:,0:yshape,2] = vis[8,:,:]

        img[xshape:,yshape:,0] = vis[9,:,:]
        img[xshape:,yshape:,1] = vis[10,:,:]
        img[xshape:,yshape:,2] = vis[11,:,:]
        '''

        print shape(img)
        mi(img);pause(0.01)#,opj(img_path,str(n)+'.png'));pause(0.01)
        if node >= 0 and node < 4:
            imsave(opj(img_path,'{}_{}.png'.format(n, node)),img)
        else:
            imsave(opj(img_path,str(n)+'.png'),img)
        print img_path



#############################
inception_layers = ['ip2']
if True:

    ## BALA: changed Karl's net setup code 
    ''' 
    #solver_name = opjh('kzpy3/caf7/z2_color/solver_temp.prototxt')
    #solver_name = opjh('caffe/kzpy3/caf7/z2_color/solver_run_cat.prototxt')
    solver_name = opjh('caffe/kzpy3/caf7/z2_color/solver_temp_bala.prototxt')
    solver = setup_solver(solver_name)
    weights_file_path = opjh('caffe/kzpy3/caf5/z2_color/z2_color.caffemodel')
    #weights_file_path = opjh('z2_color_run_cat/z2_color_run_cat_iter_33400000.caffemodel')
    solver.net.copy_from(weights_file_path)

    ## uncomment these if loading data through a conv layer, not a data layer
    #solver.net.params['data'][0].data[:] = 1
    #solver.net.params['data'][1].data[:] = 0

    model_folder = solver_name.split('/')[-2]
    '''

    MODEL_NUM = 4 # 0=z2, 1=z2_color, 2=z2_color_long, 3=z1_color, 4=z2_color_small_ip1
    print 'trying net'
    net = get_net(MODEL_NUM)
    print 'got net'
    model_folder = model_folders[MODEL_NUM]

    sess_inds = [130, 160, 200]
    sess_ind = sess_inds[0]
    input_data = get_imgs(model_folder, sess_ind) ## BALA: used as input_data

    drive_mode = 5
    one_eye = None #'right' or 'left', or None
    one_timept = None #0-9 for z2_color_long, or None

    script_dir = './caf/kaffe/y2017/'
    h5f = h5py.File(script_dir+'z1_color_ip1_pca4.h5', 'r') # pick which PC to use
    sorted_inds = np.array(h5f.get('sorted_inds'))
    unsorted_comp = np.array(h5f.get('unsorted_comp'))
    unsorted_comp = unsorted_comp*np.max(np.abs(unsorted_comp))/np.abs(unsorted_comp[sorted_inds[0]]) #MIN (0) OR MAX (-1)

    sorted_info = None #[sorted_inds, unsorted_comp] ## BALA did this

    ip2_masks =[]
    for ang in range(10):
        for mot in range(10):
            temp_mask = np.zeros(20) # custom make the ip2_mask

            temp_mask[ang] = 1
            if ang>0:
                temp_mask[ang-1] = .5
            if ang<9:
                temp_mask[ang+1] = .5

            temp_mask[10+mot] = 1
            if mot>0:
                temp_mask[10+mot-1] = .5
            if mot<9:
                temp_mask[10+mot+1] = .5

            ip2_masks.append(temp_mask)

    print net.blobs['data'].data.shape

    for l in inception_layers:
        layer_shape = net.blobs[l].data.shape


        do_it4(model_folder,'scratch/'+time_str(),l,net,1000,0,node=-1,input_data=input_data,drive_mode=drive_mode,one_eye=one_eye,one_timept=one_timept,sorted_info=sorted_info,ip2_masks=ip2_masks)
        ##BALA :changed iter_n from 300 to 500 too 1000?
