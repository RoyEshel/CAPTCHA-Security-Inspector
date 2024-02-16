// makes sure that this script interacts only with iframes containing reCAPTCHA challenges
if(window.name.startsWith('c-') && window.document.getElementById('recaptcha-token') !== null)
{
    solve_image = document.createElement('img')
    solve_image.src = chrome.runtime.getURL('images/solve_image.png')
    solve_image.style.height = '32px'
    solve_image.style.width = '32px'
    solve_image.style.verticalAlign = 'middle'

    solve_button = document.createElement('button')
    solve_button.id = 'recaptcha-solve-button'
    solve_button.title = 'Solve reCAPTCHA'
    solve_button.style.height = '48px'
    solve_button.style.width = '48px'
    solve_button.style.backgroundColor = 'white' // sets the background color of the button to white instead of the default light-gray
    solve_button.style.border = 'none' // removes the border of the button
    solve_button.style.cursor = 'pointer' // makes the pointer become a hand when hovering over this button
    solve_button.addEventListener('mouseover', growSolveImage)
    solve_button.addEventListener('click', sendRecaptchaDetails, { once: true }) // the button can only be used once per reCAPTCHA challenge
    solve_button.addEventListener('mouseout', shrinkSolveImage)
    solve_button.appendChild(solve_image)

    solve_div = document.createElement('div')
    solve_div.className = 'button-holder solve-button-holder'
    solve_div.appendChild(solve_button)

    buttons_container = document.querySelector('div.rc-buttons')
    buttons_container.appendChild(solve_div)
}


// extracts the details of the reCAPTCHA challenge
function getRecaptchaDetails()
{
    let descriptions_container = document.querySelector('div.rc-imageselect-desc, div.rc-imageselect-desc-no-canonical')
    let descriptions_amount = descriptions_container.childNodes.length
    let instructions = descriptions_container.childNodes[0].textContent
    let payload_source = document.getElementsByTagName('img')[0].src  // the url of the payload (the image of the reCAPTCHA challenge)
    let requested_object = descriptions_container.childNodes[1].textContent
    let challenge_type = instructions.startsWith('Select all images') ? '3x3' : '4x4'
    if (challenge_type === '3x3' && descriptions_amount === 3)
    {
        challenge_type = 'special 3x3'
    }
    
    return [challenge_type, requested_object, payload_source]
}

// Sends the details of the reCAPTCHA challenge to the background script
function sendRecaptchaDetails()
{
    console.log('Sending challenge details to the server')
    let reCAPTCHA_details = getRecaptchaDetails()
    let message = {
        type: 'reCAPTCHA details', 
        challenge_type: reCAPTCHA_details[0], 
        requested_object: reCAPTCHA_details[1], 
        payload_source: reCAPTCHA_details[2] 
    }

    chrome.runtime.sendMessage(message, response => handleResponse(response))
}

// In the special 3x3 reCAPTCHA type, the replaced tiles hold a 100x100 image source of themself
// Therefore, we get each of their payload sources and send to background script that will pass it to the server
function sendReplacedTiles(tile_indexes)
{
    let mini_payloads = {indexes: tile_indexes, sources: []}
    tile_indexes.forEach(index => {
        let payload_source = document.getElementsByTagName('img')[index].src
        mini_payloads.sources.push(payload_source)
    })
    let message = {
        type: 'reCAPTCHA details',
        challenge_type: 'special 3x3 - mini payloads',
        requested_object: getRecaptchaDetails()[1],
        payload_source: mini_payloads
    }

    chrome.runtime.sendMessage(message, response => handleResponse(response))
    console.log('Sending Reloaded tiles to the server')
}

// Handles the response received from the background script
async function handleResponse(response)
{
    if (response.type === 'reCAPTCHA solution')
    {
        // Get the reCAPTCHA solution from the response
        let solution = response.solution
        // Get the indexes of the tiles we need to select
        let tile_indexes = getTileIndexes(solution)
        console.log('Challenge solution: ', tile_indexes) 
        // Get the challeng type
        console.log("Finding the challenge's type")
        let challenge_type = getRecaptchaDetails()[0]

        // challenge is a regular 3x3. we send only one request to the server
        if (challenge_type === '3x3')
        {
            console.log('Challenge type is regular 3x3 grid')
            // Select the tiles
            selectTiles(tile_indexes)
            // Allows us to check if the automated response was correct (debug) 
            await new Promise(resolve => setTimeout(resolve, 1000)); 
            // Submitting the solution (even if no tiles needed to be clicked)
            submitSolution()
            // Give the verify button time to change its attributes
            await new Promise(resolve => setTimeout(resolve, 1000));
            // Check if the challenge was successfully completed
            let success = isSolved()
            if (success)
            {
                announceSuccess()
            }
            else
            {
                await changeChallenge()
                sendRecaptchaDetails()
            }
        }
        // challenge is a special 3x3. we ping-pong with the server until no image contain the requested object
        else if (challenge_type === 'special 3x3')  
        {
            console.log('Challenge type is 3x3 grid with fading tiles')
            // There are tiles that need to be clicked
            if (solution) 
            {
                // Gets the payload source of the first to-be-selected tile before it is being selected
                let old_source = document.getElementsByTagName('img')[tile_indexes[0]].src
                // Selects the tiles containing the required object
                selectTiles(tile_indexes)
                // Waiting for the clicked tiles to finish reloading a new image
                console.log('Waiting for selected tiles to reappear...')
                await letPayloadSourceChange(tile_indexes[0], old_source)
                console.log('Selected tiles reappeared')
                // Send another request to the server (containing the payloads of the replaced tiles)
                sendReplacedTiles(tile_indexes)
            }
            // No tiles contain the requested object anymore
            else 
            {
                // Submitting the solution
                submitSolution()
                // Give the verify button time to change its attributes
                await new Promise(resolve => setTimeout(resolve, 1000));
                // Check if the challenge was successfully completed
                let success = isSolved()
                if (success)
                {
                    announceSuccess()
                }
                else
                {
                    await changeChallenge()
                    sendRecaptchaDetails()
                }
            }
            
        }
        // challenge is a 4x4. we ping-pong with the server for 3-4 times until all 4x4 grids are solved
        else if (challenge_type === '4x4')  
        {
            console.log('Challenge type is 4x4 grid')
            // Select the tiles
            selectTiles(tile_indexes)
            // Allows us to check if the automated response was correct (debug) 
            await new Promise(resolve => setTimeout(resolve, 1000)); 
            // Gets the payload source of the entire 4x4 grid before it is being submitted and replaced
            let old_source = document.getElementsByTagName('img')[0].src
            // Submitting the solution to the current 4x4 grid (even if no tiles needed to be clicked)
            let verify_text = submitSolution()
            
            // Checking if there are more 4x4 grids after this one
            if (verify_text !== 'Verify')
            {
                console.log('There is another grid')
                // Waiting for the current 4x4 grid to be replaced with a new one
                await letPayloadSourceChange(0, old_source)
                // Send another request to the server (with the updated payload for the new 4x4 grid)
                sendRecaptchaDetails()
            }
            else
            {
                // It was the last 4x4 grid
                // Give the verify button time to change its attributes 
                await new Promise(resolve => setTimeout(resolve, 1000));
                // Check if the challenge was successfully completed
                let success = isSolved()
                if (success)
                {
                    announceSuccess()
                }
                else
                {
                    await changeChallenge()
                    sendRecaptchaDetails()
                }
            }
        }
    }
    // The server doesn't support the requested object
    else if (response.type === 'Error')
    {
        let requested_object = getRecaptchaDetails()[1]
        console.log(`The server doesn't support ${requested_object} yet`)
        await changeChallenge()
        sendRecaptchaDetails()
    }
}

// Converts the string solution to an array of integer indexes belonging to the tiles we need to select
function getTileIndexes(solution)
{
    if (solution === '')
    {
        return []
    }
    // '0 1 5 7 8' -> ['0', '1', '5', '7', '8']
    let tile_indexes = solution.split(' ')
    // Map a new array using JS version of list comprehension. ['0', '1', '5', '7', '8'] -> [0, 1, 5, 7, 8]
    tile_indexes = tile_indexes.map(s => parseInt(s));
    return tile_indexes
}

// Selects the tiles at the indexes it receives
function selectTiles(tile_indexes)
{
    console.log('Selecting tiles')
    tile_indexes.forEach(index => {
        // the image element inside the tile
        // when clicked its according tile gets selected
        let tile_img = document.getElementsByTagName('img')[index]
        tile_img.click()
    });
}

// Uses active waiting to halt the solver while tiles change their payload
async function letPayloadSourceChange(index, old_source)
{
    let current_source = document.getElementsByTagName('img')[index].src
    while (current_source === old_source)
    {
        // Wait a second before trying again
        await new Promise(resolve => setTimeout(resolve, 1000));
        current_source = document.getElementsByTagName('img')[index].src
    }
}

// Changes the reCAPTCHA challenge
async function changeChallenge()
{
    let reload_button = document.querySelector('button.rc-button-reload')
    reload_button.addEventListener('click', () => console.log('Changing challenge...'))
    let old_source = document.getElementsByTagName('img')[0].src 
    reload_button.click()
    await letPayloadSourceChange(0, old_source)
    console.log('Challenge changed')
}

// Clicks the verify button to submit the challenge
// Returns the text that was in the button before the click
function submitSolution()
{
    console.log('Submitting solution')
    let verify_button = document.querySelector('#recaptcha-verify-button')
    verify_text = verify_button.textContent
    verify_button.click()
    return verify_text
}

// Checks if the challenge was solved successfully
function isSolved()
{
    // When the reCAPTCHA challenge is solved successfully, the verify button's "disabled" attribute's value changes to true
    let verify_button = document.querySelector('#recaptcha-verify-button')
    return verify_button.disabled
}

// Increases the size of the solve button's image
function growSolveImage()
{
    solve_image.style.height = '40px'
    solve_image.style.width = '40px'
    solve_image.style.verticalAlign = 'middle'
}

// Decreases the size of the solve button's image
function shrinkSolveImage()
{
    solve_image.style.height = '32px'
    solve_image.style.width = '32px'
    solve_image.style.verticalAlign = 'middle'
}

async function announceSuccess()
{
    let sound = new Audio()
    sound.src = chrome.runtime.getURL('audio/tada.mp3')
    await sound.play()
    alert('reCAPTCHA solved successfully!!!')
}