import streamlit as st

st.set_page_config(page_title="Satranç Oyunu", layout="wide")

st.title("♟️ Satranç Oyunu")
st.markdown("Streamlit + HTML5 tabanlı Chessboard.js ile hazırlanmıştır.")

chess_html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    #board {
      width: 500px;
      margin: 20px auto;
    }
    .board {
      border-radius: 8px;
      box-shadow: 0 0 10px rgba(0,0,0,0.15);
    }
  </style>
  <link rel="stylesheet" href="https://unpkg.com/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" />
</head>
<body>
  <div id="board" class="board"></div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.13.4/chess.min.js"></script>
  <script src="https://unpkg.com/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
  <script>
    var board = null;
    var game = new Chess();

    function onDragStart (source, piece, position, orientation) {
      if (game.game_over() || 
          (game.turn() === 'w' && piece.search(/^b/) !== -1) ||
          (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
      }
    }

    function onDrop (source, target) {
      var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
      });

      if (move === null) return 'snapback';

      updateStatus();
    }

    function onSnapEnd () {
      board.position(game.fen());
    }

    function updateStatus () {
      var status = '';
      if (game.in_checkmate()) {
        status = 'Oyun bitti. Mat oldunuz!';
      } else if (game.in_draw()) {
        status = 'Beraberlik!';
      } else {
        status = (game.turn() === 'w' ? 'Beyaz' : 'Siyah') + ' sırası.';
        if (game.in_check()) {
          status += ' Şah çekildi!';
        }
      }
      document.getElementById('status').innerHTML = status;
    }

    var config = {
      draggable: true,
      position: 'start',
      onDragStart: onDragStart,
      onDrop: onDrop,
      onSnapEnd: onSnapEnd
    };

    board = Chessboard('board', config);
  </script>

  <div style="text-align:center; margin-top: 20px;">
    <div id="status" style="font-weight: bold; font-size: 18px;">Beyaz sırası.</div>
  </div>
</body>
</html>
"""

st.components.v1.html(chess_html, height=650)
