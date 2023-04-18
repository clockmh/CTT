class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zmh/Trxy/snapshot'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/data/zmh/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/data/zmh/GOT-10k/train_data'
        self.trackingnet_dir = '/data/zmh/trackingnet'
        self.coco_dir = '/data/zmh/coco'
        #self.lvis_dir = ''
        #self.sbd_dir = ''
        self.imagenet_dir = '/data/zmh/vid/vid'
        #self.imagenetdet_dir = ''
        #self.ecssd_dir = ''
        #self.hkuis_dir = ''
        #self.msra10k_dir = ''
        #self.davis_dir = ''
        self.youtubevos_dir = '/data/zmh/ytbb'
