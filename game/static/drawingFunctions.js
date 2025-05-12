//functions for drawing the game

//define game canvas
var canvas = document.getElementById("gameCanvas");
var ctx = document.getElementById("gameCanvas").getContext("2d");

//display parameters.
var canvasWidth = canvas.width,
  canvasHeight = canvas.height;

var playersCardHeight = 125,
  playersCardWidth = 86;
var prevCardScale = 0.8;
var prevCardHeight = prevCardScale * playersCardHeight,
  prevCardWidth = prevCardScale * playersCardWidth;
var oppCardScale = 0.8;
var oppCardHeight = oppCardScale * playersCardHeight,
  oppCardWidth = oppCardScale * playersCardWidth;

var oppCardDeltaX = 25;
var prevCardDeltaX = oppCardDeltaX,
  prevCardDeltaY = 35;
var playerCardDeltaX = (1.0 / oppCardScale) * oppCardDeltaX;
var playerSelectedDeltaY = 20;

var playerCardInitialX = 480,
  playerCardInitialY = 600;
var prevCardInitialX = 680,
  prevCardInitialY = 280;
var oppInitialX = [100, 529, 1300],
  oppInitialY = [200, 50, 200];
var oppCardDirections = [
  [0, 1],
  [1, 0],
  [0, 1],
];
var oppCardRot = [-Math.PI / 2, 0, Math.PI / 2];

var whosGoCirclePos = [
  [720, 530],
  [250, 400],
  [720, 200],
  [1200, 400],
];
var whosGoCircleRad = 20;
var whosGoColourControl = "#59e63d";
var whosGoColourNormal = "#f44e4e";

//functions to draw hands.

function drawPlayersHand() {
  for (var i = 0; i < playersHand.length; i++) {
    cardId = "pc" + (i + 1);
    card = document.getElementById(cardId);
    //card.innerHTML = ("<img src='/static/cardImages/" + String(playersHand[i]) + ".png' width='" + String(playersCardWidth) + "' height='" + String(playersCardHeight) + "'/>");
    card.innerHTML =
      "<img src='" +
      staticURL +
      "cardImages/" +
      String(playersHand[i]) +
      ".png' width='" +
      String(playersCardWidth) +
      "' height='" +
      String(playersCardHeight) +
      "'/>";
    card.style.left = String(playerCardInitialX + i * playerCardDeltaX) + "px";
    card.style.width = String(playersCardWidth) + "px";
    card.style.height = String(playersCardHeight) + "px";
    card.style.visibility = "visible";
    card.style.zIndex = String(i);
    var selected = 0;
    for (var j = 0; j < selectedHand.length; j++) {
      if (playersHand[i] == selectedHand[j]) {
        selected = 1;
      }
    }
    if (selected) {
      card.style.top = String(playerCardInitialY - playerSelectedDeltaY) + "px";
    } else {
      card.style.top = String(playerCardInitialY) + "px";
    }
  }

  for (var i = playersHand.length; i < 13; i++) {
    cardId = "pc" + (i + 1);
    card = document.getElementById(cardId);
    card.style.visibility = "hidden";
  }
}

function selectCard(n) {
  var alreadythere = 0;
  var index = 0;
  for (var i = 0; i < selectedHand.length; i++) {
    if (playersHand[n - 1] == selectedHand[i]) {
      alreadythere = 1;
      index = i;
    }
  }
  if (alreadythere) {
    //remove
    selectedHand.splice(index, 1);
  } else {
    selectedHand.push(playersHand[n - 1]);
  }
  drawPlayersHand();
}

function drawOpponentsHands() {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  for (var opp = 0; opp < 3; opp++) {
    ox = oppInitialX[opp];
    oy = oppInitialY[opp];
    for (var i = 0; i < nOppCards[opp]; i++) {
      ctx.save();
      ctx.translate(ox + oppCardWidth / 2, oy + oppCardHeight / 2);
      ctx.rotate(oppCardRot[opp]);
      ctx.translate(
        -1 * (ox + oppCardWidth / 2),
        -1 * (oy + oppCardHeight / 2)
      );
      ctx.drawImage(backOfCard, ox, oy, oppCardWidth, oppCardHeight);
      ctx.restore();
      ox += oppCardDirections[opp][0] * oppCardDeltaX;
      oy += oppCardDirections[opp][1] * oppCardDeltaX;
    }
  }
}

function drawPreviousHands() {
  var zIndCounter = 1;
  //draw three previous hands.
  for (var j = 0; j < 3; j++) {
    for (var i = 0; i < prevHands[j].length; i++) {
      var cardId = "prev" + String(j + 1) + String(i + 1);
      var card = document.getElementById(cardId);
      //card.innerHTML = ("<img src='/static/cardImages/"+String(prevHands[j][i])+".png' width='"+String(prevCardWidth)+"' height='"+String(prevCardHeight)+"'/>");
      card.innerHTML =
        "<img src='" +
        staticURL +
        "cardImages/" +
        String(prevHands[j][i]) +
        ".png' width='" +
        String(prevCardWidth) +
        "' height='" +
        String(prevCardHeight) +
        "'/>";
      card.style.width = String(prevCardWidth) + "px";
      card.style.height = String(prevCardHeight) + "px";
      card.style.left = String(prevCardInitialX + i * prevCardDeltaX) + "px";
      card.style.top = String(prevCardInitialY + j * prevCardDeltaY) + "px";
      card.style.visibility = "visible";
      card.style.zIndex = zIndCounter;
      zIndCounter++;
    }
    for (var i = prevHands[j].length; i < 5; i++) {
      var cardId = "prev" + String(j + 1) + String(i + 1);
      var card = document.getElementById(cardId);
      card.style.visibility = "hidden";
    }
  }
}

function drawWhosGo() {
  //draw a circle which signifies whose go it is and if they have control.
  if (control == 1) {
    ctx.fillStyle = whosGoColourControl;
  } else {
    ctx.fillStyle = whosGoColourNormal;
  }
  cx = whosGoCirclePos[playersGo - 1][0];
  cy = whosGoCirclePos[playersGo - 1][1];
  ctx.beginPath();
  ctx.arc(cx, cy, whosGoCircleRad, 0, 2 * Math.PI);
  ctx.fill();
}

function updateScores() {
  for (var i = 0; i < 4; i++) {
    pid = "player" + (i + 1);
    playerText = document.getElementById(pid);
    playerText.innerHTML =
      "P" + String(i + 1) + ": " + String(sessionScores[i]);
  }
}

//draw full game from scratch with current settings.
function fullDrawUpdate() {
  drawPlayersHand();
  drawOpponentsHands();
  drawPreviousHands();
  drawWhosGo();
  updateScores();

  ngc = document.getElementById("newGame");
  if (gameActive == 0) {
    ngc.style.visibility = "visible";
  } else {
    ngc.style.visibility = "hidden";
  }
}
