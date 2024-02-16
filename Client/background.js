const serverAddress = 'http://192.168.68.90:8000'

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse)
{
  if (message.type === 'manipulate reCAPTCHA') 
  {
    chrome.scripting.executeScript({
  
      target: { tabId: sender.tab.id, allFrames: true},
  
      files: ['manipulateRecaptcha.js'],
  
    });
  }
  else if (message.type === "change anchor's logo")
  {
    chrome.scripting.executeScript({
  
      target: { tabId: sender.tab.id, allFrames: true},
  
      files: ['changeAnchorLogo.js'],
  
    });
  }
  else if (message.type === 'reCAPTCHA details')
  {
    fetch(serverAddress, {
      method: 'POST',
      body: JSON.stringify({
        challenge_type: message.challenge_type, 
        requested_object: message.requested_object, 
        payload_source: message.payload_source
      })
    })
    .then(async response => {
      switch(response.statusText)
      {
        case 'OK':
          sendResponse({type: 'reCAPTCHA solution', solution: await response.text()})
          break;
        case 'Bad Request':
          sendResponse({type: 'Error', instructions: 'refresh reCAPTCHA'});
          break;
      }
    })
    .catch(error => console.error(error)) 
  }
  return true; // indicate that sendResponse will be called asynchronously
});