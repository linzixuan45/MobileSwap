# MobileSwap

The code will be released

In this work, we propose a real-time, lightweight, and high-quality facelift model called MobileSwap. Unlike most models that achieve facial exchange by designing complex network structures and loss functions, our MobileSwap can better decouple identity and background information while only using simple network structures and loss functions, thereby generating higher quality exchanged faces


    -------------------------------Simswap---------------------------------
        MODEL        FID
    0    Gnet  11.356283
    1  Gnet-F   6.604606
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.585811                     0.105856
    1  Gnet-F                     0.546471                     0.167207
    2    Base                     0.028615                     0.028615
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.483300                      0.173577
    1  Gnet-F                      0.444176                      0.242107
    2    Base                      0.022909                      0.022909
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.478514               0.142962
    1  Gnet-F               0.438620               0.203385
    2    Base               0.019192               0.019192
        MODEL       POSE
    0    Gnet  12.165376
    1  Gnet-F  13.498561

    -------------------------------Ours MobileSWap---------------------------------
        MODEL       FID
    0    Gnet  8.518880
    1  Gnet-F  6.198755
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.644234                     0.119226
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.669281                      0.088625
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.628347               0.082978
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.304962
    1  Gnet-F  1.227651
