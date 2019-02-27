
regions = {
    '78': {
        'tik_pattern': 'spb-2018-TIK-(\d+)-.*',
        'uik_pattern': 'r78_u(\d+)_(.+)',
        'src_dir': '/mnt/ftp/2018-Spb/',
        'dst_dir': '/mnt/2018-4TB-2/data/2018-Spb/',
        'tmp_dir': '/mnt/2018-4TB-2/data/tmp/78_concat/',
        'gap_file': 'gaps_78.json',
        'box_file': 'boxes_78.json',
    },
    '47': {
        'tik_pattern': 'TIK-(\w+)-.*',
        'uik_pattern': 'r47_u(\d+)_(.+)',
        'src_dir': '/mnt/OLD-4TB-2/data/2018-lenobl/',
        'dst_dir': '/mnt/2018-4TB-2/data/2018-lenobl/',
        'tmp_dir': '/mnt/2018-4TB-2/data/tmp/47_concat/',
        'gap_file': 'gaps_47.json',
        'box_file': 'boxes_47.json',
    }
} 
