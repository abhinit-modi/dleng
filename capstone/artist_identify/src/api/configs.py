""" All config for translating client and server responses """


CONTRACT_CONFIG = {
    'CLASSES': ['albrecht-durer', 'alfred-sisley', 'amedeo-modigliani', 'boris-kustodiev',
                'camille-corot', 'camille-pissarro', 'childe-hassam', 'claude-monet',
                'david-burliuk', 'edgar-degas', 'ernst-ludwig-kirchner', 'eugene-boudin',
                'francisco-goya', 'gustave-dore', 'henri-de-toulouse-lautrec', 'henri-matisse',
                'ilya-repin', 'isaac-levitan', 'ivan-aivazovsky', 'ivan-shishkin', 'james-tissot',
                'joaquã\xadn-sorolla', 'john-singer-sargent', 'konstantin-korovin',
                'konstantin-makovsky', 'marc-chagall', 'martiros-saryan', 'maurice-prendergast',
                'nicholas-roerich', 'odilon-redon', 'pablo-picasso', 'paul-cezanne', 'paul-gauguin',
                'peter-paul-rubens', 'pierre-auguste-renoir', 'pyotr-konchalovsky',
                'raphael-kirchner', 'rembrandt', 'salvador-dali', 'sam-francis', 'thomas-eakins',
                'utagawa-kuniyoshi', 'vincent-van-gogh', 'william-merritt-chase',
                'zinaida-serebriakova'],
    'IMAGE_SIZE': (224, 224),
    'RESCALE_FACTOR': 1./255
}

LOAD_TEST_CONFIG = {
    'NUM_SAMPLES': 5,
    'SOURCE_DIR': "/Users/abmodi/2020/dleng/capstone/artist_identify/data/processed/wikiart_sampled/test",
    'PERCENTILES': [.25, .5, .75, 0.95],
    'MODEL_ENDPOINT': 'http://localhost:8501/v1/models/my_baseline:predict',
}
