import React, { Component } from 'react'
import { Form, Button} from 'react-bootstrap';
import ReactPlayer from "react-player/lazy";
import logo from "./../dubly-logo-inverted.png"
import axios from 'axios';
import './Home.css';


export class Home extends Component {

    state = {
        "mode": "upload",    //upload, processing, ready
        "file": File,
        "emptyFile": true,
        "dubbedVideo": ""
    }
    
    dubVideo(e){
        if(this.state.emptyFile === true){
            alert("Please upload a video")
            return;
        }
        this.setState({ mode: "processing" })
        
        //send video to backend then change mode to ready when dubbed video is returned
        var FilmFormData = new FormData();
        FilmFormData.append("file", this.state.file[0]);
        axios.post('http://localhost:8080/translate', FilmFormData, {
            headers: {
                "Content-Type": "multipart/form-data"
            }
        }).then((res) => {
            if(res.status===200) // Successful
            {
                console.log(res.data)
                this.setState({
                    dubbedVideo: res.data.Result,
                    mode: "ready"
                })
            }
            else{
                this.setState({errorMessage: res.data.message});
                this.setState({ mode: "upload" })
            }
        })
    }

    newVideo(){
        this.setState({ mode: "upload", emptyFile: true, dubbedVideo: ""})
    }

    render(){
        return (
            <div className="Home-div container-fluid">
                    <div className={this.state.mode === "processing" ? "Processing-div":"Upload-div"}>
                        {
                            this.state.mode === "upload" ?
                            (   
                                <div>
                                    <img src={logo} style={{"width":"35%", "padding": "3% 0px 0px 0px"}} alt="logo"></img>
                                    <h1 style={{"fontFamily": "Lucida Handwriting", "padding": "8% 0px 0px 0px"}}>The Magic of video dubbing</h1>
                                    <Form>
                                        <Form.Group controlId="formFileSm" className="mb-3" >
                                            <Form.Control type="file" size="sm" onChange={e => this.setState({ file: e.target.files, emptyFile: false})}/>
                                        </Form.Group>
                                        <Button style={{"width":"150px"}} onClick={this.dubVideo.bind(this)}>Dub Video</Button>                                
                                    </Form>
                                </div>
                            ):(
                                <div></div>
                            )
                        }
                        {
                            this.state.mode === "processing" ?
                            (   
                                <div>
                                    <img src={logo} style={{"width":"15%"}} alt="logo"></img>
                                    <h1 style={{"fontFamily": "Lucida Handwriting"}}>Applying our magic</h1>
                                    <h1 style={{"fontFamily": "Lucida Handwriting"}}>Please wait!</h1>
                                </div>
                            ):(
                                <div></div>
                            )
                        }
                        {
                            this.state.mode === "ready" ?
                            (
                                <div>
                                    <img src={logo} style={{"width":"15%"}} alt="logo"></img>
                                    <h1 style={{"fontFamily": "Lucida Handwriting", "padding": "0px"}}>The Magic of video dubbing is applied</h1>
                                    <hr/>
                                    <h2 style={{"fontFamily": "Lucida Handwriting", "textAlign": "left", "padding": "0px"}}>Original Video:</h2>
                                    <ReactPlayer url={URL.createObjectURL(this.state.file[0])} width="100%" height="100%" controls id='video'/>
                                    <hr/>
                                    <h2 style={{"fontFamily": "Lucida Handwriting", "textAlign": "left", "padding": "0px"}}>Dubbed Video:</h2>
                                    <ReactPlayer url={this.state.dubbedVideo} width="100%" height="100%" controls id='video'/>
                                    <Button style={{"width":"150px"}} onClick={() => this.newVideo()}>New Video</Button> 
                                    <hr/> 
                                </div>
                            ):(
                                <div></div>
                            )
                        }
                    </div>
            </div>
        );
    }
}

export default Home;