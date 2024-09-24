# Mel-Steg-cINN

<img src="https://user-images.githubusercontent.com/55458365/147407359-25cb0fe0-5361-42bc-83f7-6454411516c0.png" alt="drawing" width="30"/> **PL**

**Autor:** Sławomir Nikiel

Niniejszy projekt stanowi kod źródłowy pracy magisterskiej pod tytułem "Sieci głębokie w zastosowaniach do ukrywania informacji w ścieżkach dźwiękowych" autorstwa absolwenta Politechniki Warszawskiej Sławomira Nikla.

***Streszczenie pracy dyplomowej***

Steganografia to zbiór metod pozwalających na prowadzenie komunikacji w sposób niezauważalny dla otoczenia. Tajne wiadomości mogą być ukrywane w różnych nośnikach informacji, takich jak obrazy lub ścieżki dźwiękowe. W przypadku ukrywania informacji w tych drugich mówimy o audio steganografii, czyli zbiorze technik służących do ukrywania informacji w nośnikach dźwięku. 

Celem niniejszej pracy jest zbadanie potencjału głębokich sieci neuronowych do ukrywania informacji w dźwięku. Wykorzystaną reprezentacją dźwięku, w której ukrywane są tajne wiadomości, jest mel spektrogram. Natomiast rolę kodera i dekodera wiadomości przyjęła warunkowa odwracalna sieć neuronowa, warunkowana siecią UNET. Dodatkowo zbadany został potencjał sieci neuronowej w kontekście odzyskiwania ukrytych informacji ze stratnie skompresowanych mel spektrogramów.

Finalnym efektem projektu magisterskiego jest rozwiązanie wykorzystujące odwracalne sieci neuronowe do ukrywania informacji w nośniku dźwięku metodą kolorowania mel spektrogramów. Wytrenowano dwa modele sieci neuronowych przystosowane do operowania na nośnikach nieskompresowanych oraz skompresowanych. Obie sieci cechuje wysoka jakość generowanych mel spektrogramów oraz wysoka dokładność odzyskiwania informacji. W przypadku modelu operującego na nieskompresowanych danych sprawność odzyskiwania informacji wynosi 100\%. Natomiast w przypadku zastosowania kompresji udało się opracować metodę pozwalającą na odzyskiwanie informacji ze sprawnością wynoszącą 99,8\%.

Poza szczegółowym opisem opracowanych architektur sieci neuronowych w pracy można znaleźć omówienie przeprowadzonych treningów, opis procesu wyboru funkcji strat oraz przedstawienie przebiegu uczenia sieci. Dodatkowo zaproponowany został projekt kompletnego systemu steganograficznego wraz z omówieniem całego procesu przetwarzania danych wykorzystywanych przez modele sieci. Praca zwieńczona jest wynikami szeregu przeprowadzonych eksperymentów obejmujących takie kwestie jak badanie jakości generowanych mel spektrogramów, badanie dokładności odzyskiwania informacji, subiektywny test odsłuchu odzyskiwanego dźwięku, czy metody poprawy skuteczności odzyskiwania wiadomości.

<img src="https://user-images.githubusercontent.com/55458365/147407290-1cc0142b-b0d3-43aa-aa65-a702e8371c7b.png" alt="drawing" width="30"/> **ENG**

**Author:** Sławomir Nikiel

This project is the source code for a master's thesis entitled "Deep networks in applications for hiding information in audio" by Warsaw University of Technology graduate Sławomir Nikiel.

***Abstract of the thesis***

Steganography is a set of methods that allow communication to be carried out in a way that is imperceptible to the observer. Secret messages can be hidden in a variety of digital media, such as images or audio tracks. In the case of hiding information in the latter, we speak of audio steganography, a set of techniques for hiding information in audio media. 

The purpose of this work is to explore the potential of deep neural networks in hiding information in sound. We use mel spectrograms as audio representation used for concealing secret messages. The conditional invertible neural network is a codec of the messages. It is conditioned by the UNET network. Additionally, the potential of the neural networks in the context of recovering hidden information from lossy compressed mel spectrograms was studied as well.

The final result of the master's thesis project is a solution that uses mel spectrogram coloring and invertible neural networks to hide information in the sound carriers. Two models of neural networks adapted to operate on uncompressed and compressed media were trained. Both networks are capable of generating high quality mel spectrograms and retriving information with high accuracy. In the case of the model operating on uncompressed data, the efficiency of information recovery is 100\%. In the case of compression usage, we succeeded in developing a method to recover information with an efficiency of 99.8\%.

In addition to a detailed description of the developed neural network architectures, the paper includes a discussion on the performed trainings, a description of the process of selecting the loss functions, and a presentation of the network learning flow. In addition, the design of a complete steganographic system is proposed, along with a discussion on the entire process of data preparation for the neural network models. The work is topped out with the results of a number of experiments that were carried out, covering such topics as testing the quality of the generated spectrograms, testing the accuracy of information recovery, subjective listening tests of the recovered audio, and methods of improving the efficiency of message recovery.

