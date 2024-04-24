import numpy as np
import cv2
import msvcrt

from keras.models import load_model

overlay = cv2.imread('BG.png', -1)
win_overlay = cv2.imread("WIN.png", -1)
lose_overlay = cv2.imread("LOSE.png", -1)

p_overlay = cv2.imread("PTURN.png", -1)
d_overlay = cv2.imread("DTURN.png", -1)

#nilai card
card_values = {
    "2-Keriting": 2, "3-Keriting": 3, "4-Keriting": 4, "5-Keriting": 5, "6-Keriting": 6, "7-Keriting": 7, "8-Keriting": 8, "9-Keriting": 9, "10-Keriting": 10, "Jack-Keriting": 10, "Queen-Keriting": 10, "King-Keriting": 10, "Ace-Keriting": 11,
     "2-Hati": 2, "3-Hati": 3, "4-Hati": 4, "5-Hati": 5, "6-Hati": 6, "7-Hati": 7, "8-Hati": 8, "9-Hati": 9, "10-Hati": 10, "Jack-Hati": 10, "Queen-Hati": 10, "King-Hati": 10, "Ace-Hati": 11,
    "2-Berlian" : 2, "3-Berlian" : 3, "4-Berlian" : 4, "5-Berlian" : 5, "6-Berlian" : 6, "7-Berlian" : 7, "8-Berlian" : 8, "9-Berlian" : 9, "10-Berlian" : 10, "Jack-Berlian" : 10, "Queen-Berlian" : 10, "King-Berlian" : 10, "Ace-Berlian" : 11,
    "2-Sekop" : 2, "3-Sekop" : 3, "4-Sekop" : 4, "5-Sekop" : 5, "6-Sekop" : 6, "7-Sekop" : 7, "8-Sekop" : 8, "9-Sekop" : 9, "10-Sekop" : 10, "Jack-Sekop" : 10, "Queen-Sekop" : 10, "King-Sekop" : 10,"Ace-Sekop" : 11
}

#nama card
LabelKelas = (
    "2-Keriting", "3-Keriting", "4-Keriting", "5-Keriting", "6-Keriting", "7-Keriting", "8-Keriting", "9-Keriting", "10-Keriting", "Jack-Keriting", "Queen-Keriting", "King-Keriting", "Ace-Keriting",
    "2-Hati", "3-Hati", "4-Hati", "5-Hati", "6-Hati", "7-Hati", "8-Hati", "9-Hati", "10-Hati", "Jack-Hati", "Queen-Hati", "King-Hati", "Ace-Hati",
    "2-Berlian", "3-Berlian", "4-Berlian", "5-Berlian", "6-Berlian", "7-Berlian", "8-Berlian", "9-Berlian", "10-Berlian", "Jack-Berlian", "Queen-Berlian", "King-Berlian", "Ace-Berlian",
    "2-Sekop", "3-Sekop", "4-Sekop", "5-Sekop", "6-Sekop", "7-Sekop", "8-Sekop", "9-Sekop", "10-Sekop", "Jack-Sekop", "Queen-Sekop", "King-Sekop","Ace-Sekop"
)

model = load_model("REMI_HASIL.h5")
detected_cards = []

def sumCard(cards):
    sum = 0
    for card in cards:
        sum += card_values[card]
    
    return sum

def detect(frame):
    CitraGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    Thresholded = cv2.adaptiveThreshold(CitraGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 10)
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(Thresholded, 4, cv2.CV_32S)
    card = []
    cards = []
    for i in range(totalLabels):
        LocKartu = values[i, 2:4]
        if (200 < LocKartu[0] < 400 and 400 < LocKartu[1] < 600):
            card.append(i)

    for i in card:
        frame = cv2.rectangle(frame, values[i, 0:2], values[i, 0:2] + values[i, 2:4], (255, 255, 0), 2, cv2.LINE_AA)

    for i in card:
        topLeft = values[i, 0:2]
        bottomRight = values[i, 0:2] + values[i, 2:4]
        cardImage = Thresholded[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        cardImage = cv2.cvtColor(cardImage, cv2.COLOR_GRAY2BGR)
        
        #save card press c
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            detected_cards.append(cardImage)
            save_card = True

            #bikin text (masi bug)
            cv2.putText(frame, 'Kartu Disimpan', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        X = []
        img = cv2.resize(cardImage, (128, 128))
        img = np.asarray(img) / 255
        img = img.astype('float32')
        X.append(img)
        X = np.array(X)
        X = X.astype('float32')
        hs = model.predict(X, verbose=0)
        n = np.argmax(hs)
        
        #take nama kelas
        predicted_label = LabelKelas[n]
        numerical_value = card_values.get(predicted_label, 0)  # take nilai kelas
        
        cards.append(predicted_label)

        #show nama sama nilai
        text = f'{predicted_label} - Value: {numerical_value}'
        text_x = topLeft[0]
        text_y = topLeft[1] - 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

    frame_resized = cv2.resize(frame, (overlay.shape[1], overlay.shape[0]))
    alpha = overlay[:, :, 3] / 255.0
    overlay_area = frame_resized.copy()

    for c in range(0, 3):
        overlay_area[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * overlay_area[:, :, c])

    #show yang di overlay
    #cv2.imshow('Webcam Overlay', overlay_area)

    if cards and len(cards) != 0:
        return cards[-1], overlay_area
    else:
        return None, overlay_area

def game():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ## Game State; 0 = start, 1 = player's turn, 2 = dealer's turn
    state = 0
    ## 0 = no winner, 1 = player, 2 = dealer
    winner = 0
    ## If Player or Dealer already stand
    playerStand = False
    dealerStand = False

    ## List Cards
    playerCard = []
    dealerCard = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break    
        
        card, img = detect(frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break

        if state == 0:
            ## START GAME
            state = 1
        elif state == 1 and not playerStand:
            alpha = p_overlay[:, :, 3] / 255.0
        
            for c in range(0, 3):
                img[:, :, c] = (alpha * p_overlay[:, :, c] + (1 - alpha) * img[:, :, c])
        
            if key == ord('1'):
                print("")
                print("PRESS SPACE TO DETECT CARD")
                while True:
                    
                    # Detect
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera error")
                        break    
                    
                    card, img = detect(frame)
                    
                    alpha = p_overlay[:, :, 3] / 255.0
                
                    for c in range(0, 3):
                        img[:, :, c] = (alpha * p_overlay[:, :, c] + (1 - alpha) * img[:, :, c])
                        
                    cv2.imshow("Game", img)
                    
                    key = cv2.waitKey(1)
                    
                    
                    if key == ord('q'):
                        break                    
                    # Check if there is keyboard is hit AND if get char is space
                    if key == 32 and card:
                        print(f'Detected Card is {card}')

                        # Append card
                        playerCard.append(card)
                        state = 2
                        break


            elif key == ord('2'):
                print("Player Stand")
                # Straightforward
                playerStand = True
                # Change turn
                state = 2

            
        elif state == 2 and not dealerStand:
            ## DEALER'S TURN

            alpha = d_overlay[:, :, 3] / 255.0
        
            for c in range(0, 3):
                img[:, :, c] = (alpha * d_overlay[:, :, c] + (1 - alpha) * img[:, :, c])

            # If keyboard press space
            if key == 32 and card:
                print(f'Detected Card is {card}')  

                print("Dealer Takes Card")
                dealerCard.append(card)
                # Change Turn
                state = 1
                
            
            # If sum of dealer's card is more than 16, then dealer stand
            # Casino's rule or something
            if sumCard(dealerCard) > 16:
                print("Dealer Stand")
                dealerStand = True


            

        ## STOP CONDITION
        # Sum of cards
        sumPlayer = sumCard(playerCard)
        sumDealer = sumCard(dealerCard)
    
        # if either has total of exactly 21
        # then winner is guarantee
        if sumPlayer == 21:
            winner = 1
        elif sumDealer == 21:
            winner = 2
    
        # if either has over 21, then they lose
        if sumPlayer > 21:
            winner = 2
        elif sumDealer > 21:
            winner = 1
    
        # If both stands
        if playerStand and dealerStand:
            # check which deck has higher value
            if sumPlayer > sumDealer:
                winner = 1
            else:
                winner = 2
        

        ## If Winner
        if winner != 0:
            if winner == 1:
                alpha = win_overlay[:, :, 3] / 255.0
            
                for c in range(0, 3):
                    img[:, :, c] = (alpha * win_overlay[:, :, c] + (1 - alpha) * img[:, :, c])
            elif winner == 2:
                alpha = lose_overlay[:, :, 3] / 255.0
            
                for c in range(0, 3):
                    img[:, :, c] = (alpha * lose_overlay[:, :, c] + (1 - alpha) * img[:, :, c])
                
        cv2.imshow("Game", img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    game()