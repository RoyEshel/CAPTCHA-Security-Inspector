// makes sure that this script interacts only with iframes containing reCAPTCHA anchors
if (window.name.startsWith('a-') && window.document.querySelector('#recaptcha-token') !== null)
{
    let logo_image = chrome.runtime.getURL('icons/icon32.png')
    let logo_div = document.querySelector('div.rc-anchor-logo-img, div.rc-anchor-logo-img-portrait')
    logo_div.style.backgroundImage = `url("${logo_image}")`

    let logo_text = document.querySelector('div.rc-anchor-logo-text')
    logo_text.textContent = 'Security Inspector'

    let parent = document.querySelector('div.rc-anchor-normal-footer')
    let pt_div = document.querySelector('div.rc-anchor-pt')
    parent.removeChild(pt_div)
}