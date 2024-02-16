const targetNode = document

const observerOptions = {
    subtree: true,   // listens for changes in all sub-trees of the document as well
    childList: true, // listens for changes in elements' childList
}

// setting an observer that listens for DOM changes in search of reCAPTCHA anchors
const anchorObserver = new MutationObserver(function(mutations) {   
    let anchorFound = false
    
    mutations.forEach(function(mutation) 
    {   
        for (let i = 0; i < mutation.addedNodes.length; i++)
        {
            if (mutation.addedNodes[i].tagName == 'IFRAME')
            {
                let addedNode = mutation.addedNodes[i]

                if (addedNode.title == 'reCAPTCHA') // the added node is the CAPTCHA anchor
                {
                    anchorFound = true 
                }
                else if (addedNode.title == 'recaptcha challenge expires in two minutes') // the added node is an iframe containing the actual reCAPTCHA challenge (with its language set to english)
                {
                    // this event listener is removed after handling its first 'focus' event by the use of the 'once' option
                    // that way each reCAPTCHA frame is manipulated only once by the 'manipulateRecaptcha.js' script
                    addedNode.addEventListener('focus', manipulateRecaptcha, { once: true })
                }
            } 
        }
    })
    
    if (anchorFound)
    {
        forceEnglish()
    }
})

forceEnglish() // catch reCAPTCA anchors which loaded prior to the observer's activation

anchorObserver.observe(targetNode, observerOptions) // activating the mutation observer


function forceEnglish() // forces the language of all reCAPTCHA test to english
{
    // getting all reCAPTCHA anchors, i.e. iframes containing captcha checkbox (the i'm not a robot checkbox)
    let anchors = document.querySelectorAll("iframe[title='reCAPTCHA']")

    for (let i = 0; i < anchors.length; i++) 
    {
        let anchor = anchors[i]
        let curLang = anchor.src.match(/hl=(.*?)&/).pop() // getting the current language of the reCAPTCHA test (example: 'iw' for hebrew)
        let newLang = 'en' // new language of the reCAPTCHA test

        if (curLang !== newLang)
        {
            anchor.src = anchor.src.replace(curLang, newLang) // changing the language reCAPTCHA test
        }
    }

    setTimeout(changeAnchorLogo, 500)
}

function manipulateRecaptcha()
{
    chrome.runtime.sendMessage({ type: 'manipulate reCAPTCHA' });
}

function changeAnchorLogo()
{
    chrome.runtime.sendMessage({ type: "change anchor's logo" });
}
