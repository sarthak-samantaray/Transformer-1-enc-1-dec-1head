sentence_pairs = [
    ('where did you put your key?', 'où est-ce tu as mis ta clé?'),
    ('you missed a spot.', 'tu as loupé une tache.'),
    ("i think we're being followed.", 'je pense que nous sommes suivis.'),
    ('i bought a cactus.', "j'ai acheté un cactus."),
    ('i have more than enough.', "j'en ai plus que marre.")
]


test_pairs = [
    # Training examples
    ('where did you put your key?', 'où est-ce tu as mis ta clé?'),
    ('you missed a spot.', 'tu as loupé une tache.'),
    ("i think we're being followed.", 'je pense que nous sommes suivis.'),
    ('i bought a cactus.', "j'ai acheté un cactus."),
    ('i have more than enough.', "j'en ai plus que marre."),
    
    # New test examples
    ("You are missing a key", "Il vous manque une clé"),
    ("I am following you", "je te suis"),
    ("Cactus is enough", "Le cactus suffit"),
]