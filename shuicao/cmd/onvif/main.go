package main

import (
	"context"
	"encoding/xml"
	"flag"
	"io"
	"log"
	"time"

	"github.com/pkg/errors"
	"github.com/use-go/onvif"
	onvifdevice "github.com/use-go/onvif/device"
	onvifmedia "github.com/use-go/onvif/media"
	onvifsdkmedia "github.com/use-go/onvif/sdk/media"
	onvifxsd "github.com/use-go/onvif/xsd/onvif"
)

func setIP(ctx context.Context, device *onvif.Device) error {
	getResp, err := device.CallMethod(onvifdevice.GetNetworkInterfaces{})
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer getResp.Body.Close()
	b, err := io.ReadAll(getResp.Body)
	if err != nil {
		return errors.Wrap(err, "")
	}
	interfaces := struct {
		Header struct{}
		Body   struct {
			GetNetworkInterfacesResponse struct {
				NetworkInterfaces struct {
					Token   onvifxsd.ReferenceToken `xml:"token,attr"`
					Enabled bool
					IPv4    struct {
						Enabled bool
						Config  struct {
							Manual struct {
								Address      string
								PrefixLength int
							}
							LinkLocal struct {
								Address      string
								PrefixLength int
							}
							FromDHCP struct {
								Address      string
								PrefixLength int
							}
							DHCP bool
						}
					}
				}
			}
		}
	}{}
	if err := xml.Unmarshal(b, &interfaces); err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("%#v", interfaces)
	ni := interfaces.Body.GetNetworkInterfacesResponse.NetworkInterfaces

	// setInterface := onvifdevice.SetNetworkInterfaces{}
	setInterface := struct {
		XMLName          string                  `xml:"tds:SetNetworkInterfaces"`
		InterfaceToken   onvifxsd.ReferenceToken `xml:"tds:InterfaceToken"`
		NetworkInterface struct {
			Enabled bool `xml:"onvif:Enabled"`
			IPv4    struct {
				Enabled bool `xml:"onvif:Enabled"`
				Manual  struct {
					Address      string `xml:"onvif:Address"`
					PrefixLength int    `xml:"onvif:PrefixLength"`
				} `xml:"onvif:Manual"`
				DHCP bool `xml:"onvif:DHCP"`
			} `xml:"onvif:IPv4"`
		} `xml:"tds:NetworkInterface"`
	}{}
	setInterface.InterfaceToken = ni.Token
	setInterface.NetworkInterface.Enabled = true
	setInterface.NetworkInterface.IPv4.Enabled = true
	setInterface.NetworkInterface.IPv4.Manual.Address = "169.254.36.6"
	setInterface.NetworkInterface.IPv4.Manual.PrefixLength = 16
	setInterface.NetworkInterface.IPv4.DHCP = true
	log.Printf("%#v", setInterface)
	setResp, err := device.CallMethod(setInterface)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer setResp.Body.Close()
	setB, err := io.ReadAll(setResp.Body)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("%s", setB)

	setZero := onvifdevice.SetZeroConfiguration{}
	setZero.InterfaceToken = ni.Token
	setZero.Enabled = true
	setZeroResp, err := device.CallMethod(setZero)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer setZeroResp.Body.Close()
	setZeroB, err := io.ReadAll(setZeroResp.Body)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("%s", setZeroB)

	getZero := onvifdevice.GetZeroConfiguration{}
	getZeroResp, err := device.CallMethod(getZero)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer getZeroResp.Body.Close()
	getZeroB, err := io.ReadAll(getZeroResp.Body)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("%s", getZeroB)

	return nil
}

func getStreamUri(ctx context.Context, device *onvif.Device) (string, error) {
	getResp, err := device.CallMethod(onvifmedia.GetProfiles{})
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	defer getResp.Body.Close()
	b, err := io.ReadAll(getResp.Body)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	log.Printf("%s", b)
	profiles := struct {
		Header struct{}
		Body   struct {
			GetProfilesResponse onvifmedia.GetProfilesResponse
		}
	}{}
	if err := xml.Unmarshal(b, &profiles); err != nil {
		return "", errors.Wrap(err, "")
	}
	log.Printf("%#v", profiles)
	profile := profiles.Body.GetProfilesResponse.Profiles[1]

	stream, err := onvifsdkmedia.Call_GetStreamUri(ctx, device, onvifmedia.GetStreamUri{ProfileToken: profile.Token})
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	log.Printf("%#v", stream)
	uri := string(stream.MediaUri.Uri)
	return uri, nil
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	xaddr := "192.168.0.123:80"
	username := "admin"
	password := "123456"
	device, err := onvif.NewDevice(onvif.DeviceParams{Xaddr: xaddr, Username: username, Password: password})
	if err != nil {
		return errors.Wrap(err, "")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := setIP(ctx, device); err != nil {
		return errors.Wrap(err, "")
	}
	return nil

	uri, err := getStreamUri(ctx, device)
	if err != nil {
		return errors.Wrap(err, "")
	}
	log.Printf("%#v", uri)

	return nil
}
